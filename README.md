
### Installation
`apt install cmake libx11-dev libopenblas-dev liblapack-dev python-opencv python-skimage`

**NOTE:** The following (`dlib` compilation) is very memory consuming!
You may want to free as much memory as possible first (eg. `systemctl stop lightdm.service`)
Also increase swap size to ~512 (`CONF_SWAPSIZE` in `/etc/dphys-swapfile`)

`pip install -r requirements.txt`

To use `face_recognition.capture` you need `pip install picamera[array]`
