# syntax=docker/dockerfile:1.7

ARG BUILDER_IMAGE
FROM ${BUILDER_IMAGE} AS wheels

ARG SOURCE_DATE_EPOCH
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY . /src/b12x
WORKDIR /src/b12x
RUN --mount=type=cache,id=lil-wheel-pip,target=/root/.cache/pip,sharing=locked \
    env -u PYTHONPATH SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
      python -m pip wheel \
      --no-build-isolation --no-deps --wheel-dir /wheelhouse . \
    && test "$(find /wheelhouse -maxdepth 1 -name 'b12x-*.whl' | wc -l)" -eq 1 \
    && env -u PYTHONPATH python -m pip install \
      --no-deps --force-reinstall /wheelhouse/b12x-*.whl \
    && env -u PYTHONPATH python -c \
      'import b12x; print("b12x_wheel_import=PASS")'

FROM scratch AS export
COPY --from=wheels /wheelhouse/ /wheels/
