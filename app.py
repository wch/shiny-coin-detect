# This app uses a phone or tablet's camera to take a picture and process it. It cannot
# use a desktop computer's webcam. If opened on a desktop computer, it will open up an
# ordinary file chooser dialog.
#
# This particular application uses some memory-intensive libraries, like skimage, and so
# it may not work properly on all phones. However, the camera input part should still
# work on most phones.

import numpy as np

from shiny import App, render, ui
from shiny.types import FileInfo, ImgData, SilentException

import coins

app_ui = ui.page_fluid(
    ui.input_file(
        "file",
        None,
        button_label="Open camera",
        # This tells it to accept still photos only (not videos).
        accept="image/*",
        # This tells it to use the phone's rear camera. Use "user" for the front camera.
        capture="environment",
    ),
    ui.output_image("image"),
)


def server(input, output, session):
    @output
    @render.image
    async def image() -> ImgData:
        file_infos: list[FileInfo] = input.file()
        if not file_infos:
            raise SilentException()

        file_info = file_infos[0]
        coins.do_detect_coins(file_info["datapath"])
        return {"src": "coins-out.jpg", "width": "100%"}


app = App(app_ui, server)
