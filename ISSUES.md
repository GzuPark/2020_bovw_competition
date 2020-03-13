## Environment

1. `SIFT_create()`

    ```sh
    cv2.error: OpenCV(4.2.0) /io/opencv_contrib/modules/xfeatures2d/src/sift.cpp:1210: error: (-213:The function/feature is not implemented) This algorithm is patented and is excluded in this configuration; Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'create'
    ```

    - **Solution**: When you use `SIFT_create()` on OpenCV, the patent is not allowed latest version, please try to version `3.4.2.17`.


2. `libgthread-2.0.so.0`

    ```sh
    ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
    ```

    - **Solution**: Try to intall `libgtk2.0-dev` library like below:
        ```sh
        sudo apt-get update && apt-get install -y libgtk2.0-dev
        ```
