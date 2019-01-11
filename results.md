

julia -L predict_survival.jl
WARNING: using DecisionTree.fit! in module compare_model conflicts with an existing identifier.
WARNING: using DecisionTree.predict in module compare_model conflicts with an existing identifier.
loaded


julia> predict_survival.run(1)
1411399 features
422 celfiles
 __10.750717 seconds__ (31.89 M allocations: 1.454 GiB, 7.95% gc time)
data ready
nrows 150 ncols 22
  __0.339536 seconds__ (500.42 k allocations: 25.283 MiB, 2.58% gc time)
LIBLINEAR.LinearModel Train accuracy: 100.00%
LIBLINEAR.LinearModel Test accuracy: 69.50%
nrows 150 ncols 400
  __0.005377 seconds__ (27 allocations: 954.688 KiB)
LIBLINEAR.LinearModel Train accuracy: 92.50%
LIBLINEAR.LinearModel Test accuracy: 81.82%

julia> predict_survival.run(2)
1411399 features
422 celfiles
  __0.451694 seconds__ (1.41 M allocations: 64.684 MiB, 11.35% gc time)
data ready
nrows 150 ncols 22
  __0.360887 seconds__ (629.67 k allocations: 31.346 MiB, 8.77% gc time)
LIBSVM.SVM{Bool} Train accuracy: 100.00%
LIBSVM.SVM{Bool} Test accuracy: 69.00%
nrows 150 ncols 400
  __0.023121 seconds__ (43 allocations: 1.230 MiB)
LIBSVM.SVM{Bool} Train accuracy: 82.00%
LIBSVM.SVM{Bool} Test accuracy: 63.64%

julia> predict_survival.run(3)
1411399 features
422 celfiles
  __0.426844 seconds__ (1.41 M allocations: 64.684 MiB, 11.34% gc time)
data ready
nrows 22 ncols 22
  __1.253575 seconds__ (2.58 M allocations: 123.696 MiB, 19.44% gc time)
DecisionTree.DecisionTreeClassifier Train accuracy: 100.00%
DecisionTree.DecisionTreeClassifier Test accuracy: 56.75%
nrows 400 ncols 400
  __0.037061 seconds__ (435 allocations: 91.125 KiB)
DecisionTree.DecisionTreeClassifier Train accuracy: 100.00%
DecisionTree.DecisionTreeClassifier Test accuracy: 54.55%

julia> predict_survival.run(4)
1411399 features
422 celfiles
  __0.412242 seconds__ (1.41 M allocations: 64.684 MiB, 7.11% gc time)
data ready
nrows 22 ncols 22
  __6.927419 seconds__ (13.71 M allocations: 812.726 MiB, 6.68% gc time)
DecisionTree.RandomForestClassifier Train accuracy: 90.91%
DecisionTree.RandomForestClassifier Test accuracy: 56.50%
nrows 400 ncols 400
  __0.017563 seconds__ (3.70 k allocations: 3.934 MiB)
DecisionTree.RandomForestClassifier Train accuracy: 96.50%
DecisionTree.RandomForestClassifier Test accuracy: 72.73%

julia> predict_survival.run(6)
1411399 features
422 celfiles
  __0.453867 seconds__ (1.41 M allocations: 64.684 MiB, 16.63% gc time)
data ready
nrows 150 ncols 22
  __4.897356 seconds__ (9.80 M allocations: 541.551 MiB, 8.40% gc time)
NaiveBayes.HybridNB Train accuracy: 100.00%
NaiveBayes.HybridNB Test accuracy: 65.75%
nrows 150 ncols 400
  __0.156508 seconds__ (232.80 k allocations: 34.072 MiB, 3.69% gc time)
NaiveBayes.HybridNB Train accuracy: 82.00%
NaiveBayes.HybridNB Test accuracy: 72.73%

julia> predict_survival.run(7)
1411399 features
422 celfiles
 __10.446106 seconds__ (31.93 M allocations: 1.456 GiB, 7.56% gc time)
data ready
nrows 150 ncols 22
2019-01-11 12:33:00.549326: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2398650000 Hz
2019-01-11 12:33:00.550838: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x77f0f10 executing computations on platform Host. Devices:
2019-01-11 12:33:00.550878: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-01-11 12:33:01.014775: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x7849070 executing computations on platform CUDA. Devices:
2019-01-11 12:33:01.014807: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2019-01-11 12:33:01.014814: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (1): GeForce GTX 960, Compute Capability 5.2
2019-01-11 12:33:01.014820: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (2): GeForce GTX 960, Compute Capability 5.2
2019-01-11 12:33:01.015819: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8475
pciBusID: 0000:05:00.0
totalMemory: 7.93GiB freeMemory: 7.20GiB
2019-01-11 12:33:01.015965: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 1 with properties: 
name: GeForce GTX 960 major: 5 minor: 2 memoryClockRate(GHz): 1.342
pciBusID: 0000:09:00.0
totalMemory: 3.95GiB freeMemory: 3.90GiB
2019-01-11 12:33:01.016097: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 2 with properties: 
name: GeForce GTX 960 major: 5 minor: 2 memoryClockRate(GHz): 1.342
pciBusID: 0000:0a:00.0
totalMemory: 3.95GiB freeMemory: 3.90GiB
2019-01-11 12:33:01.016393: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0, 1, 2
2019-01-11 12:33:01.930787: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-11 12:33:01.930821: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 1 2 
2019-01-11 12:33:01.930828: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N N N 
2019-01-11 12:33:01.930832: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 1:   N N Y 
2019-01-11 12:33:01.930836: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 2:   N Y N 
2019-01-11 12:33:01.931314: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6935 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0, compute capability: 6.1)
2019-01-11 12:33:01.931674: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 3620 MB memory) -> physical GPU (device: 1, name: GeForce GTX 960, pci bus id: 0000:09:00.0, compute capability: 5.2)
2019-01-11 12:33:01.932021: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 3620 MB memory) -> physical GPU (device: 2, name: GeForce GTX 960, pci bus id: 0000:0a:00.0, compute capability: 5.2)
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:269
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:269
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:317
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:317
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:321
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:321
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:322
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:322
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:331
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:331
2019-01-11 12:33:30.339920: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
 __47.664685 seconds__ (54.15 M allocations: 2.876 GiB, 3.06% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 100.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 69.25%
nrows 150 ncols 400
2019-01-11 12:33:45.035996: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0, 1, 2
2019-01-11 12:33:45.036108: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-11 12:33:45.036119: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 1 2 
2019-01-11 12:33:45.036126: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N N N 
2019-01-11 12:33:45.036132: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 1:   N N Y 
2019-01-11 12:33:45.036140: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 2:   N Y N 
2019-01-11 12:33:45.036521: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6935 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0, compute capability: 6.1)
2019-01-11 12:33:45.036725: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 3620 MB memory) -> physical GPU (device: 1, name: GeForce GTX 960, pci bus id: 0000:09:00.0, compute capability: 5.2)
2019-01-11 12:33:45.036928: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 3620 MB memory) -> physical GPU (device: 2, name: GeForce GTX 960, pci bus id: 0000:0a:00.0, compute capability: 5.2)
 __15.813147 seconds__ (6.73 M allocations: 2.605 GiB, 3.03% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 93.75%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 81.82%

TF_USE_GPU=0 julia -L predict_survival.jl
(v1.0) pkg> build TensorFlow
  Building Arpack ──────────→ `~/.julia/packages/Arpack/UiiMc/deps/build.log`
  Building SpecialFunctions → `~/.julia/packages/SpecialFunctions/fvheQ/deps/build.log`
  Building Rmath ───────────→ `~/.julia/packages/Rmath/Py9gH/deps/build.log`
  Building Conda ───────────→ `~/.julia/packages/Conda/uQitS/deps/build.log`
  Building MbedTLS ─────────→ `~/.julia/packages/MbedTLS/r1Ufc/deps/build.log`
  Building PyCall ──────────→ `~/.julia/packages/PyCall/0jMpb/deps/build.log`
  Building CodecZlib ───────→ `~/.julia/packages/CodecZlib/DAjXH/deps/build.log`
  Building TensorFlow ──────→ `~/.julia/dev/TensorFlow/deps/build.log`
 Resolving package versions...

julia> predict_survival.run(7)
1411399 features
422 celfiles
  __6.283147 seconds__ (23.00 M allocations: 957.196 MiB, 10.39% gc time)
data ready
nrows 150 ncols 22
2019-01-11 12:49:46.294024: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:269
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:269
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:317
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:317
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:321
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:321
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:322
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:322
┌ Warning: `set_field!(obj::Any, fld::Symbol, val)` is deprecated, use `setproperty!(obj, fld, val)` instead.
│   caller = extend_graph(::TensorFlow.Graph, ::Array{Any,1}) at core.jl:331
└ @ TensorFlow ~/.julia/dev/TensorFlow/src/core.jl:331
 __39.918413 seconds__ (44.55 M allocations: 2.331 GiB, 2.86% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 100.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 69.50%
nrows 150 ncols 400
 __18.812418 seconds__ (6.27 M allocations: 2.582 GiB, 2.59% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 93.25%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 77.27%

... second build and test differs, is there random behaviour?
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 81.82%

... with local build for opt CPU
julia> predict_survival.run(7)
1411399 features
422 celfiles
 __10.365921 seconds__ (31.81 M allocations: 1.450 GiB, 7.76% gc time)
data ready
nrows 150 ncols 22
2019-01-11 14:38:14.673303: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2398650000 Hz
2019-01-11 14:38:14.674524: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x76fa230 executing computations on platform Host. Devices:
2019-01-11 14:38:14.674574: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
 __41.811496 seconds__ (49.92 M allocations: 2.675 GiB, 2.91% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 100.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 69.25%
nrows 150 ncols 400
 __15.663253 seconds__ (7.21 M allocations: 2.628 GiB, 3.16% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 94.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 81.82%

julia> predict_survival.run(7)
1411399 features
422 celfiles
  __0.410243 seconds__ (1.41 M allocations: 64.684 MiB, 7.87% gc time)
data ready
nrows 150 ncols 22
 __13.816734 seconds__ (10.26 M allocations: 667.886 MiB, 2.91% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 100.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 69.50%
nrows 150 ncols 400
 __15.443998 seconds__ (6.72 M allocations: 2.604 GiB, 3.02% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 93.75%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 81.82%

... rerun gpu
julia> predict_survival.run(7)
1411399 features
422 celfiles
 __10.553316 seconds__ (31.81 M allocations: 1.450 GiB, 7.82% gc time)
data ready
nrows 150 ncols 22
2019-01-11 15:31:09.005297: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2398650000 Hz
2019-01-11 15:31:09.006357: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x2677650 executing computations on platform Host. Devices:
2019-01-11 15:31:09.006380: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
2019-01-11 15:31:09.480709: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x7091e00 executing computations on platform CUDA. Devices:
2019-01-11 15:31:09.480756: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): GeForce GTX 1080, Compute Capability 6.1
2019-01-11 15:31:09.480768: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (1): GeForce GTX 960, Compute Capability 5.2
2019-01-11 15:31:09.480777: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (2): GeForce GTX 960, Compute Capability 5.2
2019-01-11 15:31:09.482265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 0 with properties: 
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.8475
pciBusID: 0000:05:00.0
totalMemory: 7.93GiB freeMemory: 7.20GiB
2019-01-11 15:31:09.482487: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 1 with properties: 
name: GeForce GTX 960 major: 5 minor: 2 memoryClockRate(GHz): 1.342
pciBusID: 0000:09:00.0
totalMemory: 3.95GiB freeMemory: 3.90GiB
2019-01-11 15:31:09.482687: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1433] Found device 2 with properties: 
name: GeForce GTX 960 major: 5 minor: 2 memoryClockRate(GHz): 1.342
pciBusID: 0000:0a:00.0
totalMemory: 3.95GiB freeMemory: 3.90GiB
2019-01-11 15:31:09.483096: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0, 1, 2
2019-01-11 15:31:10.301876: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-11 15:31:10.301908: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 1 2 
2019-01-11 15:31:10.301914: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N N N 
2019-01-11 15:31:10.301918: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 1:   N N Y 
2019-01-11 15:31:10.301922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 2:   N Y N 
2019-01-11 15:31:10.302360: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6934 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0, compute capability: 6.1)
2019-01-11 15:31:10.302717: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 3620 MB memory) -> physical GPU (device: 1, name: GeForce GTX 960, pci bus id: 0000:09:00.0, compute capability: 5.2)
2019-01-11 15:31:10.303026: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 3620 MB memory) -> physical GPU (device: 2, name: GeForce GTX 960, pci bus id: 0000:0a:00.0, compute capability: 5.2)
2019-01-11 15:31:37.446954: I tensorflow/stream_executor/dso_loader.cc:152] successfully opened CUDA library libcublas.so.10.0 locally
 __47.237315 seconds__ (50.12 M allocations: 2.684 GiB, 2.84% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 100.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 69.00%
nrows 150 ncols 400
2019-01-11 15:31:53.115398: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0, 1, 2
2019-01-11 15:31:53.115507: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-11 15:31:53.115519: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 1 2 
2019-01-11 15:31:53.115526: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N N N 
2019-01-11 15:31:53.115531: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 1:   N N Y 
2019-01-11 15:31:53.115536: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 2:   N Y N 
2019-01-11 15:31:53.115922: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6934 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0, compute capability: 6.1)
2019-01-11 15:31:53.116123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 3620 MB memory) -> physical GPU (device: 1, name: GeForce GTX 960, pci bus id: 0000:09:00.0, compute capability: 5.2)
2019-01-11 15:31:53.116344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 3620 MB memory) -> physical GPU (device: 2, name: GeForce GTX 960, pci bus id: 0000:0a:00.0, compute capability: 5.2)
 __17.049616 seconds__ (6.74 M allocations: 2.605 GiB, 3.99% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 93.25%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 77.27%

julia> predict_survival.run(7)
1411399 features
422 celfiles
  __0.439741 seconds__ (1.41 M allocations: 64.684 MiB, 12.30% gc time)
data ready
nrows 150 ncols 22
2019-01-11 15:32:16.770275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0, 1, 2
2019-01-11 15:32:16.770369: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-11 15:32:16.770380: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 1 2 
2019-01-11 15:32:16.770386: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N N N 
2019-01-11 15:32:16.770392: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 1:   N N Y 
2019-01-11 15:32:16.770397: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 2:   N Y N 
2019-01-11 15:32:16.770779: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6934 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0, compute capability: 6.1)
2019-01-11 15:32:16.770970: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 3620 MB memory) -> physical GPU (device: 1, name: GeForce GTX 960, pci bus id: 0000:09:00.0, compute capability: 5.2)
2019-01-11 15:32:16.771156: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 3620 MB memory) -> physical GPU (device: 2, name: GeForce GTX 960, pci bus id: 0000:0a:00.0, compute capability: 5.2)
 __16.785832 seconds__ (10.22 M allocations: 665.579 MiB, 2.58% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 100.00%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 69.50%
nrows 150 ncols 400
2019-01-11 15:32:34.008334: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1512] Adding visible gpu devices: 0, 1, 2
2019-01-11 15:32:34.008442: I tensorflow/core/common_runtime/gpu/gpu_device.cc:984] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-11 15:32:34.008453: I tensorflow/core/common_runtime/gpu/gpu_device.cc:990]      0 1 2 
2019-01-11 15:32:34.008460: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 0:   N N N 
2019-01-11 15:32:34.008466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 1:   N N Y 
2019-01-11 15:32:34.008482: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1003] 2:   N Y N 
2019-01-11 15:32:34.008856: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6934 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1080, pci bus id: 0000:05:00.0, compute capability: 6.1)
2019-01-11 15:32:34.009051: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:1 with 3620 MB memory) -> physical GPU (device: 1, name: GeForce GTX 960, pci bus id: 0000:09:00.0, compute capability: 5.2)
2019-01-11 15:32:34.009185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:2 with 3620 MB memory) -> physical GPU (device: 2, name: GeForce GTX 960, pci bus id: 0000:0a:00.0, compute capability: 5.2)
 __16.755544 seconds__ (6.72 M allocations: 2.604 GiB, 3.05% gc time)
Main.predict_survival.compare_model.TensorFlowClassifier Train accuracy: 93.75%
Main.predict_survival.compare_model.TensorFlowClassifier Test accuracy: 81.82%


