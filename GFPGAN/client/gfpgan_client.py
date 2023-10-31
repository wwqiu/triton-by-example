import sys
import cv2
import numpy as np
import tritonclient.http as httpclient


def normalize(input_tensor, mean, std, inplace=False):
    if not inplace:
        input_tensor = input_tensor.copy()
    dtype = input_tensor.dtype
    mean = np.asarray(mean, dtype=dtype)
    std = np.asarray(std, dtype=dtype)
    mean = mean.reshape((1, -1, 1, 1))
    std = std.reshape((1, -1, 1, 1))
    input_tensor -= mean
    input_tensor /= std
    return input_tensor


if __name__ == "__main__":
    try:
        triton_client = httpclient.InferenceServerClient(
            url="localhost:8000", verbose=False
        )
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    img = cv2.imread("cropped_faces/10045_00.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input = np.expand_dims(img / 255.0, axis=0).transpose(0, 3, 1, 2)
    normalize(input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    input = input.astype(np.float32)
    inputs = [httpclient.InferInput("INPUT__0", input.shape, "FP32")]
    inputs[0].set_data_from_numpy(input)
    outputs = [httpclient.InferRequestedOutput("OUTPUT__0")]

    respone = triton_client.infer(
        "gfpgan",
        inputs,
        request_id=str(1),
        outputs=outputs,
    )

    output0 = respone.as_numpy("OUTPUT__0")
    output0 = np.clip(output0, -1, 1)
    output0 = ((output0.transpose(0, 2, 3, 1) + 1) / 2 * 255).astype(np.uint8)
    output0 = output0.squeeze()
    output0 = cv2.cvtColor(output0, cv2.COLOR_BGR2RGB)
    cv2.imwrite("restored_face.png", output0)
