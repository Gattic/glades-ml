mkdir build
cd build

CMAKE_ARGS=""
for arg in "$@"; do
	case "$arg" in
		cuda|--cuda)
			CMAKE_ARGS="$CMAKE_ARGS -DGLADES_ENABLE_CUDA=ON"
			;;
	esac
done

cmake .. $CMAKE_ARGS
make
