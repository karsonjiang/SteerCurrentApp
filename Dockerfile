FROM python:3.12-slim

# The Gmsh Python wheel loads OpenGL and X11 shared libraries even headlessly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libgomp1 libstdc++6 zlib1g \
    libx11-6 libxcursor1 libxext6 libxfixes3 libxft2 libxinerama1 libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN python -c "import gmsh; import steercurrent"

RUN useradd --create-home appuser
USER appuser

ENV PORT=10000
EXPOSE 10000
CMD ["python", "-m", "steercurrent", "--host", "0.0.0.0"]
