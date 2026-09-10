/* Included by _storage.c. File payloads are never mapped or retained between runs. */
#include <fcntl.h>
#include <time.h>
#ifdef B12X_HAVE_LIBURING
#include <liburing.h>
#endif

#define PLE_BLOCK 4096u
#define PLE_READ_MAX 65536u
#define PLE_CAPSULE "b12x.ple_reader"

typedef struct {
    int fd;
    dev_t device;
    ino_t inode;
    int64_t bytes;
} ple_file_t;

typedef struct {
    int64_t shard;
    int64_t offset;
    size_t file;
    bool scale;
} ple_source_t;

typedef struct {
    size_t file;
    int64_t offset;
    unsigned length;
    char *destination;
} ple_fragment_t;

typedef struct {
    size_t file, begin, end;
    int64_t offset;
    unsigned length, expected;
} ple_job_t;

typedef struct ple_reader ple_reader_t;
typedef struct {
    char *buffer;
    size_t job;
} ple_slot_t;

struct ple_reader {
    int64_t shard_rows, padded_rows, tp_start, tp_end;
    size_t weight_bytes, scale_bytes, max_lookups, capacity;
    bool poisoned;
    pthread_mutex_t api_mutex;
    unsigned slots_count;
    ple_slot_t *slots;
    char *buffers;
    ple_fragment_t *fragments;
    ple_job_t *jobs;
    size_t fragment_count, job_count;
    ple_file_t *files;
    ple_source_t *sources;
    size_t file_count, source_count;
    failure_t failure;
    uint64_t lookups, requested_bytes, read_bytes, read_calls;
    uint64_t unique_blocks, coalesced_reads, submit_calls;
    double execution_seconds;
#ifdef B12X_HAVE_LIBURING
    struct io_uring ring;
    bool ring_ready;
    struct iovec *iovecs;
    unsigned *free_slots;
#endif
};

#ifdef B12X_HAVE_LIBURING
static void ple_error(ple_reader_t *reader, const char *operation, int error) {
    if (!reader->failure.message[0])
        snprintf(reader->failure.message, sizeof(reader->failure.message), "%s: %s",
                 operation, strerror(error));
}

static bool ple_scatter(ple_reader_t *reader, size_t index, char *buffer, int result) {
    ple_job_t *job = &reader->jobs[index];
    reader->read_calls++;
    if (result > 0) reader->read_bytes += (unsigned)result;
    if (result < 0) {
        ple_error(reader, "io_uring READ_FIXED", -result);
        return false;
    }
    if ((unsigned)result != job->expected) {
        if (!reader->failure.message[0])
            snprintf(reader->failure.message, sizeof(reader->failure.message),
                     "short PLE read at offset %lld: expected %u bytes, received %d; source changed or was truncated",
                     (long long)job->offset, job->expected, result);
        return false;
    }
    for (size_t i = job->begin; i < job->end; i++) {
        ple_fragment_t *fragment = &reader->fragments[i];
        memcpy(fragment->destination, buffer + (fragment->offset - job->offset),
               fragment->length);
    }
    return true;
}

static void ple_release(ple_reader_t *reader) {
    if (!reader) return;
    if (reader->ring_ready) io_uring_queue_exit(&reader->ring);
    free(reader->iovecs);
    free(reader->free_slots);
    for (size_t i = 0; i < reader->file_count; i++) close(reader->files[i].fd);
    free(reader->files);
    free(reader->sources);
    free(reader->buffers);
    free(reader->slots);
    free(reader->fragments);
    free(reader->jobs);
    pthread_mutex_destroy(&reader->api_mutex);
    free(reader);
}

static void ple_delete(PyObject *capsule) {
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (reader) {
        Py_BEGIN_ALLOW_THREADS
        ple_release(reader);
        Py_END_ALLOW_THREADS
    }
}
#endif

static PyObject *py_ple_reader(PyObject *self, PyObject *args) {
    (void)self;
    long long shard_rows, padded_rows, tp_start, tp_end, weight_bytes, scale_bytes, max_lookups;
    int depth;
    if (!PyArg_ParseTuple(args, "LLLLLLLi", &shard_rows, &padded_rows,
                          &tp_start, &tp_end, &weight_bytes, &scale_bytes, &max_lookups,
                          &depth)) return NULL;
#ifndef B12X_HAVE_LIBURING
    return PyErr_Format(PyExc_RuntimeError,
                        "io_uring PLE support is unavailable: install liburing development headers and pkg-config, then rebuild the b12x native loader");
#else
    if (shard_rows <= 0 || padded_rows <= 0 || tp_start < 0 || tp_end < tp_start ||
        tp_end > padded_rows || weight_bytes <= 0 || scale_bytes < 0 || max_lookups <= 0 ||
        depth <= 0 || depth > 32768 ||
        weight_bytes > PY_SSIZE_T_MAX / max_lookups ||
        scale_bytes > PY_SSIZE_T_MAX / max_lookups || max_lookups > PY_SSIZE_T_MAX / 8)
        return PyErr_Format(PyExc_ValueError, "invalid PLE reader geometry or I/O capacity");
    /* At most ceil((row_bytes + block - 1) / block) fragments per plane. */
    size_t per_row = (size_t)weight_bytes / PLE_BLOCK + 2;
    if (scale_bytes) per_row += (size_t)scale_bytes / PLE_BLOCK + 2;
    if ((size_t)max_lookups > SIZE_MAX / per_row ||
        (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_fragment_t) ||
        (size_t)max_lookups * per_row > SIZE_MAX / sizeof(ple_job_t))
        return PyErr_Format(PyExc_OverflowError, "PLE batch metadata is too large");
    ple_reader_t *reader = calloc(1, sizeof(*reader));
    if (!reader) return PyErr_NoMemory();
    /* Only an initialized API mutex reaches ple_release. */
    int error = pthread_mutex_init(&reader->api_mutex, NULL);
    if (error) { free(reader); return PyErr_Format(PyExc_RuntimeError, "PLE mutex: %s", strerror(error)); }
    reader->shard_rows = shard_rows;
    reader->padded_rows = padded_rows;
    reader->tp_start = tp_start;
    reader->tp_end = tp_end;
    reader->weight_bytes = weight_bytes;
    reader->scale_bytes = scale_bytes;
    reader->max_lookups = max_lookups;
    reader->capacity = (size_t)max_lookups * per_row;
    reader->slots_count = (unsigned)depth;
    reader->slots = calloc(reader->slots_count, sizeof(*reader->slots));
    reader->fragments = calloc(reader->capacity, sizeof(*reader->fragments));
    reader->jobs = calloc(reader->capacity, sizeof(*reader->jobs));
    error = posix_memalign((void **)&reader->buffers, PLE_BLOCK,
                          (size_t)reader->slots_count * PLE_READ_MAX);
    if (!reader->slots || !reader->fragments || !reader->jobs || error) {
        ple_release(reader);
        return PyErr_NoMemory();
    }
    for (unsigned i = 0; i < reader->slots_count; i++) {
        reader->slots[i].buffer = reader->buffers + (size_t)i * PLE_READ_MAX;
    }
    error = io_uring_queue_init(reader->slots_count, &reader->ring, 0);
    if (error < 0) {
        ple_release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring initialization failed: %s; enable io_uring in the kernel/container policy",
                            strerror(-error));
    }
    reader->ring_ready = true;
    reader->iovecs = calloc(reader->slots_count, sizeof(*reader->iovecs));
    reader->free_slots = calloc(reader->slots_count, sizeof(*reader->free_slots));
    if (!reader->iovecs || !reader->free_slots) { ple_release(reader); return PyErr_NoMemory(); }
    for (unsigned i = 0; i < reader->slots_count; i++) {
        reader->iovecs[i].iov_base = reader->slots[i].buffer;
        reader->iovecs[i].iov_len = PLE_READ_MAX;
    }
    error = io_uring_register_buffers(&reader->ring, reader->iovecs, reader->slots_count);
    if (error < 0) {
        ple_release(reader);
        return PyErr_Format(PyExc_RuntimeError,
                            "io_uring buffer registration failed: %s; raise the locked-memory limit or reduce queue_depth",
                            strerror(-error));
    }
    PyObject *capsule = PyCapsule_New(reader, PLE_CAPSULE, ple_delete);
    if (!capsule) ple_release(reader);
    return capsule;
#endif
}

static int ple_source_compare(const void *left, const void *right) {
    const ple_source_t *a = left, *b = right;
    if (a->shard != b->shard) return a->shard < b->shard ? -1 : 1;
    return (int)a->scale - (int)b->scale;
}

static PyObject *py_ple_reader_add(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule;
    const char *path;
    long long shard, offset;
    int scale;
    if (!PyArg_ParseTuple(args, "OLsLp", &capsule, &shard, &path, &offset, &scale)) return NULL;
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (!reader) return NULL;
    if (shard < 0 || shard > (reader->padded_rows - 1) / reader->shard_rows || offset < 0 ||
        (scale && !reader->scale_bytes))
        return PyErr_Format(PyExc_ValueError, "invalid PLE source shard, offset, or scale plane");
    int64_t first = shard * reader->shard_rows;
    int64_t rows = reader->padded_rows - first;
    if (rows > reader->shard_rows) rows = reader->shard_rows;
    size_t row_bytes = scale ? reader->scale_bytes : reader->weight_bytes;
    if ((uint64_t)rows > (uint64_t)(INT64_MAX - offset) / row_bytes)
        return PyErr_Format(PyExc_OverflowError, "PLE source byte range overflows int64");
    if (first >= reader->tp_end || first + rows <= reader->tp_start) Py_RETURN_NONE;
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->api_mutex);
    ple_source_t key = {.shard = shard, .scale = scale != 0};
    if (reader->source_count && bsearch(&key, reader->sources, reader->source_count,
                                        sizeof(key), ple_source_compare)) {
        snprintf(failure.message, sizeof(failure.message), "PLE source shard/plane is already registered");
    } else {
        /* Reject FIFOs via fstat without blocking waiting for a writer. */
        int fd = open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_DIRECT);
        struct stat status;
        if (fd < 0) system_error(&failure, "open PLE source");
        else if (fstat(fd, &status) != 0) system_error(&failure, "fstat PLE source");
        else if (!S_ISREG(status.st_mode) || offset + rows * (int64_t)row_bytes > status.st_size)
            snprintf(failure.message, sizeof(failure.message), "PLE source range exceeds a regular file's size");
        else {
            size_t file = 0;
            while (file < reader->file_count &&
                   (reader->files[file].device != status.st_dev || reader->files[file].inode != status.st_ino)) file++;
            ple_source_t *sources = realloc(reader->sources, (reader->source_count + 1) * sizeof(*sources));
            if (!sources) snprintf(failure.message, sizeof(failure.message), "could not allocate PLE source descriptor");
            else {
                reader->sources = sources;
                if (file == reader->file_count) {
                    ple_file_t *files = realloc(reader->files, (file + 1) * sizeof(*files));
                    if (!files) snprintf(failure.message, sizeof(failure.message), "could not allocate PLE file descriptor");
                    else {
                        reader->files = files;
                        reader->files[file] = (ple_file_t){fd, status.st_dev, status.st_ino, status.st_size};
                        reader->file_count++;
                        fd = -1;
                    }
                } else if (reader->files[file].bytes != status.st_size) {
                    snprintf(failure.message, sizeof(failure.message), "PLE source file size changed during registration");
                }
                if (!failure.message[0]) {
                    reader->sources[reader->source_count++] = (ple_source_t){shard, offset, file, scale != 0};
                    qsort(reader->sources, reader->source_count, sizeof(*sources), ple_source_compare);
                }
            }
        }
        if (fd >= 0) close(fd);
    }
    pthread_mutex_unlock(&reader->api_mutex);
    Py_END_ALLOW_THREADS
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
}

static int ple_fragment_compare(const void *left, const void *right) {
    const ple_fragment_t *a = left, *b = right;
    if (a->file != b->file) return a->file < b->file ? -1 : 1;
    return a->offset < b->offset ? -1 : a->offset > b->offset;
}

/* libc qsort may allocate a merge buffer on every run. Heap-sort in place to
 * keep all batch metadata persistent, including for duplicate-heavy batches. */
static void ple_sift(ple_fragment_t *items, size_t root, size_t count) {
    ple_fragment_t value = items[root];
    while (root < count / 2) {
        size_t child = root * 2 + 1;
        if (child + 1 < count && ple_fragment_compare(&items[child], &items[child + 1]) < 0)
            child++;
        if (ple_fragment_compare(&value, &items[child]) >= 0) break;
        items[root] = items[child];
        root = child;
    }
    items[root] = value;
}

static void ple_sort(ple_fragment_t *items, size_t count) {
    for (size_t i = count / 2; i > 0; i--) ple_sift(items, i - 1, count);
    for (size_t end = count; end > 1; end--) {
        ple_fragment_t value = items[end - 1];
        items[end - 1] = items[0];
        items[0] = value;
        ple_sift(items, 0, end - 1);
    }
}

static bool ple_plan(ple_reader_t *reader, const char *ids, char *weights, char *scales, size_t count) {
    reader->fragment_count = reader->job_count = 0;
    for (size_t i = 0; i < count; i++) {
        int64_t id;
        memcpy(&id, ids + i * sizeof(id), sizeof(id));
        if (id < reader->tp_start || id >= reader->tp_end) {
            memset(weights + i * reader->weight_bytes, 0, reader->weight_bytes);
            if (reader->scale_bytes) memset(scales + i * reader->scale_bytes, 0, reader->scale_bytes);
            continue;
        }
        for (int plane = 0; plane < (reader->scale_bytes ? 2 : 1); plane++) {
            ple_source_t key = {.shard = id / reader->shard_rows, .scale = plane != 0};
            ple_source_t *source = reader->source_count ? bsearch(&key, reader->sources,
                reader->source_count, sizeof(key), ple_source_compare) : NULL;
            if (!source) {
                snprintf(reader->failure.message, sizeof(reader->failure.message),
                         "missing PLE %s source for shard %lld", plane ? "scale" : "weight", (long long)key.shard);
                return false;
            }
            size_t bytes = plane ? reader->scale_bytes : reader->weight_bytes;
            char *destination = (plane ? scales : weights) + i * bytes;
            int64_t offset = source->offset + (id % reader->shard_rows) * (int64_t)bytes;
            reader->requested_bytes += bytes;
            while (bytes) {
                unsigned length = PLE_BLOCK - (uint64_t)offset % PLE_BLOCK;
                if (bytes < length) length = (unsigned)bytes;
                reader->fragments[reader->fragment_count++] =
                    (ple_fragment_t){source->file, offset, length, destination};
                offset += length;
                destination += length;
                bytes -= length;
            }
        }
    }
    ple_sort(reader->fragments, reader->fragment_count);
    size_t cursor = 0;
    while (cursor < reader->fragment_count) {
        size_t begin = cursor;
        ple_fragment_t *fragment = &reader->fragments[cursor];
        size_t file = fragment->file;
        int64_t first = fragment->offset & ~(int64_t)(PLE_BLOCK - 1);
        int64_t last = first;
        reader->unique_blocks++;
        cursor++;
        while (cursor < reader->fragment_count) {
            fragment = &reader->fragments[cursor];
            int64_t block = fragment->offset & ~(int64_t)(PLE_BLOCK - 1);
            if (fragment->file != file || (block != last &&
                (block - last != PLE_BLOCK || block - first >= PLE_READ_MAX))) break;
            if (block != last) { reader->unique_blocks++; last = block; }
            cursor++;
        }
        unsigned aligned = (unsigned)(last - first) + PLE_BLOCK;
        int64_t available = reader->files[file].bytes - first;
        unsigned expected = available < aligned ? (unsigned)available : aligned;
        reader->jobs[reader->job_count++] = (ple_job_t){file, begin, cursor, first, aligned, expected};
        if (last != first) reader->coalesced_reads++;
    }
    return true;
}

#ifdef B12X_HAVE_LIBURING
/* Prepare a whole free-slot wave before entering the kernel. A short positive
 * submit accounts for exactly that many SQEs; retry only the remaining entries.
 * A failed submit may leave pending SQEs, so stop issuing, drain every submitted
 * CQE, and retire the ring without submitting those pending entries. */
static void ple_submit_wave(ple_reader_t *reader, unsigned free_count,
                            size_t *next, unsigned *outstanding) {
    size_t remaining = reader->job_count - *next;
    unsigned count = remaining < free_count ? (unsigned)remaining : free_count;
    for (unsigned i = 0; i < count; i++) {
        unsigned slot = reader->free_slots[i];
        struct io_uring_sqe *sqe = io_uring_get_sqe(&reader->ring);
        if (!sqe) {
            ple_error(reader, "io_uring SQ capacity", ENOSPC);
            reader->poisoned = true;
            return;
        }
        ple_job_t *job = &reader->jobs[*next];
        reader->slots[slot].job = (*next)++;
        io_uring_prep_read_fixed(sqe, reader->files[job->file].fd,
                                reader->slots[slot].buffer, job->length,
                                job->offset, (int)slot);
        io_uring_sqe_set_data64(sqe, slot);
    }
    unsigned pending = count;
    while (pending) {
        reader->submit_calls++;
        int result = io_uring_submit(&reader->ring);
        if (result == -EINTR) continue;
        if (result <= 0) {
            ple_error(reader, "io_uring submit", result < 0 ? -result : EIO);
            reader->poisoned = true;
            return;
        }
        *outstanding += (unsigned)result;
        pending -= (unsigned)result;
    }
}

static void ple_uring_run(ple_reader_t *reader) {
    unsigned outstanding = 0;
    size_t next = 0;
    for (unsigned i = 0; i < reader->slots_count; i++) reader->free_slots[i] = i;
    ple_submit_wave(reader, reader->slots_count, &next, &outstanding);
    while (outstanding) {
        struct io_uring_cqe *cqe = NULL;
        /* wait/peek do not submit pending SQEs when draining a failed submit. */
        int result = io_uring_wait_cqe(&reader->ring, &cqe);
        if (result < 0) {
            if (result != -EINTR && result != -EAGAIN) {
                ple_error(reader, "io_uring completion wait", -result);
                reader->poisoned = true;
            }
            continue;
        }
        unsigned free_count = 0;
        for (;;) {
            unsigned slot = (unsigned)io_uring_cqe_get_data64(cqe);
            int status = cqe->res;
            io_uring_cqe_seen(&reader->ring, cqe);
            outstanding--;
            ple_scatter(reader, reader->slots[slot].job, reader->slots[slot].buffer, status);
            reader->free_slots[free_count++] = slot;
            if (!outstanding) break;
            result = io_uring_peek_cqe(&reader->ring, &cqe);
            if (result < 0) {
                if (result != -EAGAIN && result != -EINTR) {
                    ple_error(reader, "io_uring completion peek", -result);
                    reader->poisoned = true;
                }
                break;
            }
        }
        /* Do not prepare refill SQEs until every collected completion has been
         * checked. An I/O error therefore leaves no speculative refill entries. */
        if (!reader->failure.message[0] && next < reader->job_count)
            ple_submit_wave(reader, free_count, &next, &outstanding);
    }
    if (reader->poisoned) {
        io_uring_queue_exit(&reader->ring);
        reader->ring_ready = false;
    }
}
#endif

static bool ple_overlap(const Py_buffer *a, size_t a_bytes, const Py_buffer *b, size_t b_bytes) {
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
    return a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes);
}

static PyObject *py_ple_reader_run(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *capsule, *ids_object, *weights_object, *scales_object;
    long long count;
    if (!PyArg_ParseTuple(args, "OOOOL", &capsule, &ids_object, &weights_object, &scales_object, &count)) return NULL;
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (!reader) return NULL;
    if (count < 0 || (uint64_t)count > reader->max_lookups)
        return PyErr_Format(PyExc_ValueError, "PLE lookup count exceeds batch capacity");
    Py_buffer ids = {0}, weights = {0}, scales = {0};
    if (PyObject_GetBuffer(ids_object, &ids, PyBUF_CONTIG_RO) < 0) return NULL;
    if (PyObject_GetBuffer(weights_object, &weights, PyBUF_CONTIG) < 0) goto failed;
    if (reader->scale_bytes && PyObject_GetBuffer(scales_object, &scales, PyBUF_CONTIG) < 0) goto failed;
    if (!reader->scale_bytes && scales_object != Py_None) {
        PyErr_SetString(PyExc_ValueError, "PLE reader has no scale plane; pass None");
        goto failed;
    }
    size_t id_bytes = (size_t)count * 8;
    size_t weight_bytes = (size_t)count * reader->weight_bytes;
    size_t scale_bytes = (size_t)count * reader->scale_bytes;
    if ((size_t)ids.len < id_bytes || (size_t)weights.len < weight_bytes || (size_t)scales.len < scale_bytes ||
        !PyBuffer_IsContiguous(&ids, 'C') || !PyBuffer_IsContiguous(&weights, 'C') ||
        (reader->scale_bytes && !PyBuffer_IsContiguous(&scales, 'C'))) {
        PyErr_SetString(PyExc_ValueError, "PLE buffers must be C-contiguous and cover count rows (IDs are native signed int64 bytes)");
        goto failed;
    }
    if (ple_overlap(&ids, id_bytes, &weights, weight_bytes) ||
        ple_overlap(&ids, id_bytes, &scales, scale_bytes) ||
        ple_overlap(&weights, weight_bytes, &scales, scale_bytes)) {
        PyErr_SetString(PyExc_ValueError, "PLE IDs and destination byte ranges must not overlap");
        goto failed;
    }
    failure_t failure = {{0}};
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->api_mutex);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    reader->failure.message[0] = 0;
    reader->lookups = count;
    reader->requested_bytes = reader->read_bytes = reader->read_calls = 0;
    reader->unique_blocks = reader->coalesced_reads = reader->submit_calls = 0;
    if (reader->poisoned)
        snprintf(reader->failure.message, sizeof(reader->failure.message), "PLE io_uring reader is unusable after a submission/completion failure; create a new reader");
    else if (ple_plan(reader, ids.buf, weights.buf, scales.buf, count) && reader->job_count) {
#ifdef B12X_HAVE_LIBURING
        ple_uring_run(reader);
#endif
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    reader->execution_seconds = (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) * 1e-9;
    failure = reader->failure;
    pthread_mutex_unlock(&reader->api_mutex);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&ids);
    PyBuffer_Release(&weights);
    if (scales.obj) PyBuffer_Release(&scales);
    if (failure.message[0]) return PyErr_Format(PyExc_RuntimeError, "%s", failure.message);
    Py_RETURN_NONE;
failed:
    PyBuffer_Release(&ids);
    if (weights.obj) PyBuffer_Release(&weights);
    if (scales.obj) PyBuffer_Release(&scales);
    return NULL;
}

static PyObject *py_ple_reader_stats(PyObject *self, PyObject *capsule) {
    (void)self;
    ple_reader_t *reader = PyCapsule_GetPointer(capsule, PLE_CAPSULE);
    if (!reader) return NULL;
    uint64_t lookups, requested, bytes, calls, blocks, coalesced, submits, staging, metadata;
    double seconds;
    Py_BEGIN_ALLOW_THREADS
    pthread_mutex_lock(&reader->api_mutex);
    lookups = reader->lookups;
    requested = reader->requested_bytes;
    bytes = reader->read_bytes;
    calls = reader->read_calls;
    blocks = reader->unique_blocks;
    coalesced = reader->coalesced_reads;
    submits = reader->submit_calls;
    seconds = reader->execution_seconds;
    staging = (uint64_t)reader->slots_count * PLE_READ_MAX;
    metadata = sizeof(*reader) + reader->capacity * (sizeof(*reader->fragments) + sizeof(*reader->jobs)) +
        reader->slots_count * sizeof(*reader->slots) + reader->source_count * sizeof(*reader->sources) +
        reader->file_count * sizeof(*reader->files);
#ifdef B12X_HAVE_LIBURING
    metadata += reader->slots_count * (sizeof(*reader->iovecs) + sizeof(*reader->free_slots));
    if (reader->ring_ready) {
        metadata += reader->ring.sq.ring_sz;
        if (reader->ring.cq.ring_ptr != reader->ring.sq.ring_ptr) metadata += reader->ring.cq.ring_sz;
        metadata += reader->ring.sq.ring_entries * sizeof(struct io_uring_sqe);
    }
#endif
    pthread_mutex_unlock(&reader->api_mutex);
    Py_END_ALLOW_THREADS
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:d}",
        "lookups", (unsigned long long)lookups, "requested_bytes", (unsigned long long)requested,
        "read_bytes", (unsigned long long)bytes, "read_calls", (unsigned long long)calls,
        "unique_blocks", (unsigned long long)blocks, "coalesced_reads", (unsigned long long)coalesced,
        "submit_calls", (unsigned long long)submits,
        "staging_bytes", (unsigned long long)staging, "metadata_bytes", (unsigned long long)metadata,
        "execution_seconds", seconds);
}
