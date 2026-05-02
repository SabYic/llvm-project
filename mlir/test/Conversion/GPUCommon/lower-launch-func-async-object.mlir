// RUN: mlir-opt %s --gpu-to-llvm -split-input-file | FileCheck %s

// Regression test for gpu.launch_func lowering with an explicit async object.
//
// In `gpu-to-llvm`, when `gpu.launch_func` carries both async dependencies and
// an explicit async object, the explicit async object must remain the launch
// stream. Async dependencies lowered to streams must instead be converted to
// events recorded on those streams, and the explicit async object must wait on
// those events before launching the kernel.
//
// The bug was that `gpu.launch_func` lowering always reused the first async
// dependency stream as the primary launch stream. As a result, an explicit
// async object was ignored, and the launch happened on the wrong stream.
module attributes {gpu.container_module} {
  gpu.module @kernel_module [#nvvm.target] {
    llvm.func @kernel() attributes {gpu.kernel} {
      llvm.return
    }
  }

  // CHECK-LABEL: @launch_with_async_object
  func.func @launch_with_async_object(%stream : !llvm.ptr) {
    %c1 = arith.constant 1 : index
    // CHECK: %[[dep:.*]] = llvm.call @mgpuStreamCreate()
    %dep = gpu.wait async
    // CHECK: %[[event:.*]] = llvm.call @mgpuEventCreate()
    // CHECK: llvm.call @mgpuEventRecord(%[[event]], %[[dep]])
    // CHECK: llvm.call @mgpuStreamWaitEvent(%arg0, %[[event]])
    // CHECK: gpu.launch_func <%arg0 : !llvm.ptr> @kernel_module::@kernel
    %t = gpu.launch_func async [%dep] <%stream : !llvm.ptr> @kernel_module::@kernel
        blocks in (%c1, %c1, %c1)
        threads in (%c1, %c1, %c1)
    gpu.wait [%t]
    return
  }
}
