# Use a newer official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    git \
    libboost-all-dev \
    autoconf \
    automake \
    autotools-dev \
    zlib1g-dev \
    libbz2-dev \
    gfortran \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh
ENV PATH="/miniconda/bin:${PATH}"

# Create a new conda environment
RUN conda create -n myenv python=3.10 -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install packages using conda
RUN conda install -c conda-forge -y \
    numpy=1.23.5 \
    scipy=1.9.3 \
    pandas=1.5.2 \
    matplotlib=3.6.2 \
    seaborn=0.12.2 \
    biopython=1.81 \
    MDAnalysis \
    mdanalysis \
    requests \
    hole2

# Install mdahole2 from GitHub
RUN pip install git+https://github.com/MDAnalysis/mdahole2.git

# Verify mdahole2 installation
RUN python -c "import mdahole2; print('mdahole2 imported successfully')"

# Install pyali from GitHub
RUN git clone https://github.com/christang/pyali.git \
    && cd pyali \
    && python setup.py install \
    && cd .. \
    && rm -rf pyali

# Install tmtools using pip
RUN pip install tmtools

# Copy DSSP to the Docker image and install it
COPY hssp-3.1.4.tar.gz /tmp/hssp.tar.gz
RUN tar -xzvf /tmp/hssp.tar.gz -C /tmp \
    && cd /tmp/hssp-3.1.4 \
    && ./autogen.sh \
    && ./configure \
    && sed -i 's/-Werror//g' Makefile \
    && make CXXFLAGS="-g -O2 -std=c++11 -pedantic -Wall -Wno-reorder" \
    && make install \
    && cd / \
    && rm -rf /tmp/hssp-3.1.4 /tmp/hssp.tar.gz \
    && if [ -f /usr/local/bin/mkhssp ]; then ln -s /usr/local/bin/mkhssp /usr/local/bin/mkdssp; fi \
    && ln -s /usr/local/bin/mkdssp /usr/local/bin/dssp

# Ensure conda environment is in the PATH
ENV PATH="/miniconda/envs/myenv/bin:$PATH"

# Fr-TM-Align setup with shorter paths
RUN mkdir -p /app/R1 /app/P_structs /home/software/frtmalign
COPY frtmalign_201307.zip /tmp/frtmalign.zip
RUN unzip /tmp/frtmalign.zip -d /home/software/frtmalign \
    && cd /home/software/frtmalign/frtmalign_201307/frtmalign \
    && make \
    && mv frtmalign /home/software/frtmalign/ \
    && cd /home/software/frtmalign \
    && rm -rf frtmalign_201307 \
    && chmod 755 frtmalign \
    && rm /tmp/frtmalign.zip

# Copy necessary files with shorter paths
COPY struct_info191031.xml simple2.rad /app/R1/
COPY paths.txt main.py frtmalign_2_msa.py frtmalign_2_msa_additional_logging.py frtmalign_2_msa_holeanalysis.py /app/

# Create a verification script
RUN echo '#!/bin/bash' > /app/verify_paths.sh && \
    echo 'echo "Verifying paths..."' >> /app/verify_paths.sh && \
    echo 'paths=(' >> /app/verify_paths.sh && \
    echo '    "/app/R1/"' >> /app/verify_paths.sh && \
    echo '    "/app/R1/struct_info191031.xml"' >> /app/verify_paths.sh && \
    echo '    "/home/software/frtmalign/frtmalign"' >> /app/verify_paths.sh && \
    echo '    "/app/P_structs/"' >> /app/verify_paths.sh && \
    echo '    "/app/R1/simple2.rad"' >> /app/verify_paths.sh && \
    echo '    "/usr/local/bin/mkdssp"' >> /app/verify_paths.sh && \
    echo '    "/app/paths.txt"' >> /app/verify_paths.sh && \
    echo '    "/app/main.py"' >> /app/verify_paths.sh && \
    echo '    "/app/frtmalign_2_msa.py"' >> /app/verify_paths.sh && \
    echo '    "/app/frtmalign_2_msa_additional_logging.py"' >> /app/verify_paths.sh && \
    echo '    "/app/frtmalign_2_msa_holeanalysis.py"' >> /app/verify_paths.sh && \
    echo ')' >> /app/verify_paths.sh && \
    echo 'for path in "${paths[@]}"; do' >> /app/verify_paths.sh && \
    echo '    if [ -e "$path" ]; then' >> /app/verify_paths.sh && \
    echo '        echo "$path exists"' >> /app/verify_paths.sh && \
    echo '    else' >> /app/verify_paths.sh && \
    echo '        echo "$path does not exist"' >> /app/verify_paths.sh && \
    echo '    fi' >> /app/verify_paths.sh && \
    echo 'done' >> /app/verify_paths.sh && \
    echo 'if command -v hole &> /dev/null; then' >> /app/verify_paths.sh && \
    echo '    echo "HOLE is installed and in PATH"' >> /app/verify_paths.sh && \
    echo 'else' >> /app/verify_paths.sh && \
    echo '    echo "HOLE is not found in PATH"' >> /app/verify_paths.sh && \
    echo 'fi' >> /app/verify_paths.sh && \
    echo 'echo "Verification complete."' >> /app/verify_paths.sh && \
    chmod +x /app/verify_paths.sh

# Define environment variables
ENV PATHS_FILE=/app/paths.txt

# Set the default command
CMD ["/bin/bash", "-c", "/app/verify_paths.sh && conda run --no-capture-output -n myenv python /app/main.py /app/paths.txt --logging detailed"]

# # Use a newer official Python runtime as a parent image
# FROM python:3.10-slim

# # Set the working directory in the container
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     wget \
#     build-essential \
#     git \
#     libboost-all-dev \
#     autoconf \
#     automake \
#     autotools-dev \
#     zlib1g-dev \
#     libbz2-dev \
#     gfortran \
#     unzip \
#     && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
#     && bash /miniconda.sh -b -p /miniconda \
#     && rm /miniconda.sh
# ENV PATH="/miniconda/bin:${PATH}"

# # Create a new conda environment
# RUN conda create -n myenv python=3.10 -y
# SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# # Install packages using conda
# RUN conda install -c conda-forge -y \
#     numpy=1.23.5 \
#     scipy=1.9.3 \
#     pandas=1.5.2 \
#     matplotlib=3.6.2 \
#     seaborn=0.12.2 \
#     biopython=1.81 \
#     MDAnalysis \
#     mdanalysis \
#     requests \
#     hole2

# # Install mdahole2 from GitHub
# RUN pip install git+https://github.com/MDAnalysis/mdahole2.git

# # Verify mdahole2 installation
# RUN python -c "import mdahole2; print('mdahole2 imported successfully')"

# # Install pyali from GitHub
# RUN git clone https://github.com/christang/pyali.git \
#     && cd pyali \
#     && python setup.py install \
#     && cd .. \
#     && rm -rf pyali

# # Install tmtools using pip
# RUN pip install tmtools

# # Copy DSSP to the Docker image and install it
# COPY hssp-3.1.4.tar.gz /tmp/hssp.tar.gz
# RUN tar -xzvf /tmp/hssp.tar.gz -C /tmp \
#     && cd /tmp/hssp-3.1.4 \
#     && ./autogen.sh \
#     && ./configure \
#     && sed -i 's/-Werror//g' Makefile \
#     && make CXXFLAGS="-g -O2 -std=c++11 -pedantic -Wall -Wno-reorder" \
#     && make install \
#     && cd / \
#     && rm -rf /tmp/hssp-3.1.4 /tmp/hssp.tar.gz \
#     && if [ -f /usr/local/bin/mkhssp ]; then ln -s /usr/local/bin/mkhssp /usr/local/bin/mkdssp; fi \
#     && ln -s /usr/local/bin/mkdssp /usr/local/bin/dssp

# # Ensure conda environment is in the PATH
# ENV PATH="/miniconda/envs/myenv/bin:$PATH"

# # Fr-TM-Align setup
# RUN mkdir -p /home/TRP-channels/Run1 /home/TRP-channels/0_provided_structures /home/software/frtmalign
# COPY frtmalign_201307.zip /tmp/frtmalign.zip
# RUN unzip /tmp/frtmalign.zip -d /home/software/frtmalign \
#     && cd /home/software/frtmalign/frtmalign_201307/frtmalign \
#     && make \
#     && mv frtmalign /home/software/frtmalign/ \
#     && cd /home/software/frtmalign \
#     && rm -rf frtmalign_201307 \
#     && chmod 755 frtmalign \
#     && rm /tmp/frtmalign.zip

# # Copy necessary files
# COPY struct_info191031.xml simple2.rad /home/TRP-channels/Run1/
# COPY paths.txt main.py frtmalign_2_msa.py frtmalign_2_msa_additional_logging.py frtmalign_2_msa_holeanalysis.py /app/

# # Create a verification script
# RUN echo '#!/bin/bash' > /app/verify_paths.sh && \
#     echo 'echo "Verifying paths..."' >> /app/verify_paths.sh && \
#     echo 'paths=(' >> /app/verify_paths.sh && \
#     echo '    "/home/TRP-channels/Run1/"' >> /app/verify_paths.sh && \
#     echo '    "/home/TRP-channels/Run1/struct_info191031.xml"' >> /app/verify_paths.sh && \
#     echo '    "/home/software/frtmalign/frtmalign"' >> /app/verify_paths.sh && \
#     echo '    "/home/TRP-channels/0_provided_structures/"' >> /app/verify_paths.sh && \
#     echo '    "/home/TRP-channels/Run1/simple2.rad"' >> /app/verify_paths.sh && \
#     echo '    "/usr/local/bin/mkdssp"' >> /app/verify_paths.sh && \
#     echo '    "/app/paths.txt"' >> /app/verify_paths.sh && \
#     echo '    "/app/main.py"' >> /app/verify_paths.sh && \
#     echo '    "/app/frtmalign_2_msa.py"' >> /app/verify_paths.sh && \
#     echo '    "/app/frtmalign_2_msa_additional_logging.py"' >> /app/verify_paths.sh && \
#     echo '    "/app/frtmalign_2_msa_holeanalysis.py"' >> /app/verify_paths.sh && \
#     echo ')' >> /app/verify_paths.sh && \
#     echo 'for path in "${paths[@]}"; do' >> /app/verify_paths.sh && \
#     echo '    if [ -e "$path" ]; then' >> /app/verify_paths.sh && \
#     echo '        echo "$path exists"' >> /app/verify_paths.sh && \
#     echo '    else' >> /app/verify_paths.sh && \
#     echo '        echo "$path does not exist"' >> /app/verify_paths.sh && \
#     echo '    fi' >> /app/verify_paths.sh && \
#     echo 'done' >> /app/verify_paths.sh && \
#     echo 'if command -v hole &> /dev/null; then' >> /app/verify_paths.sh && \
#     echo '    echo "HOLE is installed and in PATH"' >> /app/verify_paths.sh && \
#     echo 'else' >> /app/verify_paths.sh && \
#     echo '    echo "HOLE is not found in PATH"' >> /app/verify_paths.sh && \
#     echo 'fi' >> /app/verify_paths.sh && \
#     echo 'echo "Verification complete."' >> /app/verify_paths.sh && \
#     chmod +x /app/verify_paths.sh

# # Define environment variables
# ENV PATHS_FILE=/app/paths.txt

# # Set the default command
# CMD ["/bin/bash", "-c", "/app/verify_paths.sh && conda run --no-capture-output -n myenv python /app/main.py /app/paths.txt --logging detailed"]