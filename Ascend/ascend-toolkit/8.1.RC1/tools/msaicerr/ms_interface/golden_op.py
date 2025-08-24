import os
import sys
from ms_interface import utils
from ms_interface.single_op_test_frame.common.ascend_tbe_op import AscendOpKernel, AscendOpKernelRunner
import numpy as np
from te import tik


class GoldenOp:
    @staticmethod
    def golden_op(soc_version, kernel_name):
        try:
            from tbe.common import platform as cce
            from tbe.common.platform import set_current_compile_soc_info as te_set_version
        except ImportError as e:
            utils.print_warn_log("failed to import te or tbe to compile op golden op, skipped it. error:", e)
            return 
        te_set_version(soc_version)
        m = 256
        n = 32
        k = 128
        tik_instance = tik.Tik()

        input_a = tik_instance.Tensor("float16", (m, n), name="a",  scope=tik.scope_gm)
        input_b = tik_instance.Tensor("float16", (n, k), name="b",  scope=tik.scope_gm)

        input_a_ub = tik_instance.Tensor("float16", (m, n), name="a_ub",  scope=tik.scope_ubuf)
        input_b_ub = tik_instance.Tensor("float16", (m, n), name="b_ub",  scope=tik.scope_ubuf)
        output_ub = tik_instance.Tensor("float32", (m, k), name="out_ub",  scope=tik.scope_ubuf)
        output_gm = tik_instance.Tensor("float32", (m, k), name="out_ub",  scope=tik.scope_gm)

        index_scalar = tik_instance.Scalar("int32", init_value=0)
        tik_instance.data_move(input_a_ub, input_a[index_scalar], 0, 1, 32, 0, 0)
        tik_instance.data_move(input_b_ub, input_b, 0, 1, 32, 0, 0)
        tik_instance.vabs(64, output_ub, output_ub, 1, 1, 1, 8, 8)
        tik_instance.data_move(output_gm, output_ub, 0, 1, 1, 0, 0)

        kernel_name = kernel_name.replace('-', '_')
        tik_instance.BuildCCE(kernel_name,
                              inputs=[input_a, input_b],
                              outputs=[output_gm])

    @staticmethod
    def run_golden_op(soc_version, kernel_name="golden_op", device_id=0):
        # compile golden op
        try:
            utils.print_debug_log("Start to compile golden op. Please wait...")
            GoldenOp.golden_op(soc_version, kernel_name=kernel_name)
        except Exception as e:
            import traceback
            traceback.print_exc()
            utils.print_warn_log("compile golden op failed, skip golden op test.")
            return False

        kernel_name = kernel_name.replace('-', '_')
        find_path_cmd = ["find", "./kernel_meta", "-name", f"{kernel_name}*"]
        regexp = r"([_\-/0-9a-zA-Z.]{1,}\.json|[_\-/0-9a-zA-Z.]{1,}\.o|[_\-/0-9a-zA-Z.]{1,}\.cce)"
        kernel_file_list = utils.get_inquire_result(find_path_cmd, regexp)
        if not kernel_file_list:
            utils.print_warn_log(f"The {kernel_name} file path cannot be found.")

        # find golden op kernel
        for file in kernel_file_list:
            if file.endswith(".o"):
                bin_path = file
            elif file.endswith(".json"):
                json_path = file
            else:
                continue
        if not os.path.exists(bin_path) or not os.path.exists(json_path):
            utils.print_warn_log(f"Can not find bin_file  and json_file ")  
            return False
        
        # run golden op
        op_kernel = AscendOpKernel(bin_path, json_path)
        input_a = np.ones((256, 32), dtype=np.float16) * 2
        input_b = np.ones((32, 128), dtype=np.float16) * 6
        output_info = {"size": 131072, "dtype": "float32", "shape": (256, 128), "name": "output"}
        utils.print_debug_log("Start to run golden op. Please wait...")
        with AscendOpKernelRunner(device_id=device_id) as runner:
            runner.run(op_kernel, inputs=[input_a, input_b], actual_output_info=(output_info,))
        
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1]:
        SOC_VERSION = "Ascend310"
    else:
        SOC_VERSION = sys.argv[1]
    
    if len(sys.argv) < 3 or not sys.argv[2]:
        KERNEL_NAME = f"golden_op_{SOC_VERSION}"
    else:
        KERNEL_NAME = sys.argv[2]

    if len(sys.argv) < 4 or not sys.argv[3]:
        DEV_ID = 0
    else:
        try:
            DEV_ID = int(sys.argv[3])
        except ValueError:
            utils.print_warn_log("DEV_ID should be an integer")
            DEV_ID = 0
    
    RESULT = GoldenOp.run_golden_op(SOC_VERSION, kernel_name=KERNEL_NAME, device_id=DEV_ID)
    if RESULT:
        utils.print_debug_log("Run golden op successfully.")
        sys.exit(0)
    else:
        utils.print_error_log("Run golden op failed.")
        sys.exit(-1)