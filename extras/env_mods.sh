mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/host-compiler.sh" <<'EOF'
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export PATH="$CONDA_PREFIX/bin:$PATH"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/host-compiler.sh" <<'EOF'
unset CC
unset CXX
unset CUDAHOSTCXX
# PATH cleanup is handled by conda
EOF


