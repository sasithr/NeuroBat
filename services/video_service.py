"""
Video upload/storage helpers for NeuroBat.
"""

import os
import uuid

from werkzeug.utils import secure_filename

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "mov",
    "avi",
    "webm",
    "mkv",
}


def allowed_video(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_VIDEO_EXTENSIONS


def save_temporary_upload(uploaded_file, uploads_dir):
    """
    Save an uploaded video under a generated temporary filename.

    Returns:
        (original_filename, temporary_path)
    """
    original_filename = secure_filename(uploaded_file.filename)

    extension = original_filename.rsplit(".", 1)[1].lower()

    temporary_filename = (
        "upload_"
        + uuid.uuid4().hex[:12]
        + "."
        + extension
    )

    temporary_path = os.path.join(
        uploads_dir,
        temporary_filename,
    )

    uploaded_file.save(temporary_path)

    return original_filename, temporary_path


def remove_file_safely(path):
    if not path:
        return

    if not os.path.exists(path):
        return

    try:
        os.remove(path)
    except OSError:
        pass
