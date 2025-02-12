# Getting dlib face recognition to work on Cloudflare Workers
## Requirements
Make sure you've got the submodules set:
```sh
git submodule init
git submodule update
```
That will pull `dlib` and `emsdk`.

Pull and activate latest `emsdk`.
```sh
git pull
./emsdk install latest
./emsdk activate latest
```

## Set up
Source `emsdk`
```sh
source ./deps/emsdk/emsdk_env.sh
```

## Compile code using dlib.
`src/main.cpp` imports and uses dlib. To compile it to WASM do
```
em++ -std=c++17 -O3 -DDLIB_NO_GUI_SUPPORT -include ./src/force_char_traits_unsigned_int.hpp -I ./deps/dlib ./deps/dlib/dlib/all/source.cpp -lstdc++ -s MODULARIZE=1 -s EXPORT_ES6=1 -s ALLOW_MEMORY_GROWTH=1 -s USE_ZLIB=1 -s ASSERTIONS=1 --bind -O3 -g  -o dist/build/facerec.js src/main.cpp
```
