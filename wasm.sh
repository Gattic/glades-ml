mkdir build
cd build
emcmake cmake ..
emmake make -j4
emmake make install

mkdir MLDataObjects
cd MLDataObjects
emar x "../Backend/Machine Learning/DataObjects/libDataObjects.a"
cd ..
echo "Extracted DataObjects Objects"

mkdir MLGMath
cd MLGMath
emar x "../Backend/Machine Learning/GMath/libGMath.a"
cd ..
echo "Extracted GMath Objects"

mkdir MLNetworks
cd MLNetworks
emar x "../Backend/Machine Learning/Networks/libNetworks.a"
cd ..
echo "Extracted Networks Objects"

mkdir MLState
cd MLState
emar x "../Backend/Machine Learning/State/libMLState.a"
cd ..
echo "Extracted ML State Objects"

mkdir MLStructure
cd MLStructure
emar x "../Backend/Machine Learning/Structure/libMLStructure.a"
cd ..
echo "Extracted ML Structure Objects"

mkdir MLObjects
cd MLObjects
emar x "../Backend/Machine Learning/libML.a"
cd ..
echo "Extracted ML Objects"

mkdir GladesObjects
cd GladesObjects
emar x ../libglades.a
cd ..
echo "Extracted G Objects"

emar rcs libglades_combined.a GladesObjects/*.o MLObjects/*.o MLDataObjects/*.o MLGMath/*.o MLNetworks/*.o MLState/*.o MLStructure/*.o
echo "Combined G Objects into libglades_combined.a"

cp libglades_combined.a $HOME/.local/lib/libglades.a
echo "Copied libglades_combined.a to $HOME/.local/lib/libglades.a"

rm ~/.local/lib/libglades.so
echo "Removed ~/.local/lib/libglades.so (if it exists, not allowed to exist for emsdk)"
