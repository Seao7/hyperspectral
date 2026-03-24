import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def save_HSD_with_header(file_path: str, data: np.ndarray, header: bytes) -> None:
    """
    Save HSD data and header to a file.

    Args:
        file_path: Destination file path
        data: HSD data to save
        header: File header
    """
    data = data.transpose(0, 2, 1)
    data_bytes = data.tobytes()
    with open(file_path, 'wb') as file:
        file.write(header + data_bytes)


def read_HSD_from_file(file_path: str, band: int = 141):
    """
    Read an HSD file.

    Args:
        file_path: Path to the file to read
        band: Number of bands (default: 141)

    Returns:
        HSD data, header, Y-axis size, X-axis size
    """
    file_extension = file_path.split('.')[-1]
    with open(file_path, 'rb') as file:
        buffer = file.read()
    
    read_functions = {
        370623040: read_HSC180X,
        87630400: read_HSC170X_old,
        44315200: read_HSC170X_new,
        585755200: read_HSC180X_CL
    }
    
    if file_extension in ['hsd', 'dat']:
        read_func = read_functions.get(len(buffer))
        if read_func:
            hsd_data, header, Y, X = read_func(buffer)
        else:
            raise ValueError(f"Unsupported file size: {len(buffer)} bytes.")
    else:
        raise ValueError("Unsupported file extension.")
    
    return hsd_data.transpose(0, 2, 1), header, Y, X


def read_HSC180X_CL(buffer):
    """
    Read buffer in HSC180X_CL format.

    Args:
        buffer: Byte buffer to read

    Returns:
        HSD data, header, Y-axis size, X-axis size
    """
    X, Y, Z = 1920, 1080, 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    print("Header Size:", len(header), "bytes")
    dat = np.frombuffer(buffer[len(header):], dtype=np.uint16)
    dat = (dat >> 2).astype(np.uint8)
    hsd_data = np.reshape(dat, (Y, Z, X))
    return hsd_data, header, Y, X


def read_HSC180X(buffer):
    """
    Read buffer in HSC180X format.

    Args:
        buffer: Byte buffer to read

    Returns:
        HSD data, header, Y-axis size, X-axis size
    """
    X, Y, Z = 1280, 1024, 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    print("Header Size:", len(header), "bytes")
    dat = np.frombuffer(buffer[len(header):], dtype=np.uint16)
    hsd_data = np.reshape(dat, (Y, Z, X))
    return hsd_data, header, Y, X


def read_HSC170X_old(buffer):
    """
    Read buffer in old HSC170X format.

    Args:
        buffer: Byte buffer to read

    Returns:
        HSD data, header, Y-axis size, X-axis size
    """
    X, Y, Z = 640, 480, 141
    RAW_len = X * Y * Z * 2
    header = buffer[:len(buffer) - RAW_len]
    print("Header Size:", len(header), "bytes")
    dat = np.frombuffer(buffer[len(header):], dtype=np.uint16)
    hsd_data = np.reshape(dat.astype(np.uint8), (Y, Z, X))
    return hsd_data, header, Y, X


def read_HSC170X_new(buffer):
    """
    Read buffer in new HSC170X format.

    Args:
        buffer: Byte buffer to read

    Returns:
        HSD data, header, X-axis size, Y-axis size
    """
    X, Y, Z = 640, 480, 141
    RAW_len = X * Y * Z
    header = buffer[:len(buffer) - RAW_len]
    print("Header Size:", len(header), "bytes")
    dat = np.frombuffer(buffer[len(header):], dtype=np.uint8)
    hsd_data = np.reshape(dat, (Y, Z, X))
    return hsd_data, header, X, Y


def HSD_to_RGB_save(HSD: np.ndarray, file_name: str = 'untitle', use_range: int = 3, R_band: int = 55, G_band: int = 35, B_band: int = 23, gamma: float = 2.2) -> None:
    """
    Convert HSD data to an RGB image and save it.

    Args:
        HSD: Input HSD data
        file_name: Output file name (default: 'untitle')
        use_range: Range for averaging around each band (default: 3)
        R_band: Band index for red (default: 55)
        G_band: Band index for green (default: 35)
        B_band: Band index for blue (default: 23)
        gamma: Gamma correction value (default: 2.2)
    """
    RGB_array = np.stack([
        np.mean(HSD[:, :, R_band - use_range:R_band + use_range], axis=2),
        np.mean(HSD[:, :, G_band - use_range:G_band + use_range], axis=2),
        np.mean(HSD[:, :, B_band - use_range:B_band + use_range], axis=2)
    ], axis=-1)

    # Normalize to 0–255
    RGB_array = RGB_array - np.min(RGB_array)
    if np.max(RGB_array) > 0:
        RGB_array = 255 * (RGB_array / np.max(RGB_array))

    # Gamma correction
    RGB_array = (RGB_array / 255) ** (1 / gamma) * 255

    # Cast to uint8 and return
    RGB_array = RGB_array.astype('uint8')
    
    Image.fromarray(RGB_array).save(f"{file_name}.jpg")


def HSD_to_RGB(HSD: np.ndarray, use_range: int = 3, R_band: int = 55, G_band: int = 35, B_band: int = 23, gamma: float = 2.2) -> np.ndarray:
    """
    Convert HSD data to an RGB image and return the array.

    Args:
        HSD: Input HSD data (shape: (Y, X, Z))
        use_range: Range for averaging around each band (default: 3)
        R_band: Band index for red (default: 55)
        G_band: Band index for green (default: 35)
        B_band: Band index for blue (default: 23)
        gamma: Gamma correction value (default: 2.2)

    Returns:
        RGB image as uint8 array
    """
    # Compute mean over specified band ranges for each color and build RGB array
    RGB_array = np.stack([
        np.mean(HSD[:, :, R_band - use_range:R_band + use_range], axis=2),
        np.mean(HSD[:, :, G_band - use_range:G_band + use_range], axis=2),
        np.mean(HSD[:, :, B_band - use_range:B_band + use_range], axis=2)
    ], axis=-1)

    # Normalize to 0–255
    RGB_array = RGB_array - np.min(RGB_array)
    if np.max(RGB_array) > 0:
        RGB_array = 255 * (RGB_array / np.max(RGB_array))

    # Gamma correction
    RGB_array = (RGB_array / 255) ** (1 / gamma) * 255

    # Cast to uint8 and return
    RGB_array = RGB_array.astype('uint8')
    return RGB_array


def main(file_path: str, output_image_name: str) -> None:
    """
    Main processing function.

    Args:
        file_path: Input HSD file path
        output_image_name: Output image file name
    """
    hsd_data, _, _, _ = read_HSD_from_file(file_path)
    HSD_to_RGB_save(hsd_data, file_name=output_image_name)

