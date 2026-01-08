import pytest
import os

import sen2sr_tools.get_sr_image as get_sr_image

LAT = 40.0
LON = -3.0
START_DATE = "2024-01-01"
END_DATE = "2024-01-16"
BANDS = ["B04", "B08"]
GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[LAT, LON], [LAT+0.01, LON+0.01], [LAT, LON]]],
    "CRS": "epsg:4258"
    }

MOCK_SR_FILEPATH = "sen2sr_tools/SR_2025-01-01.png"


@pytest.mark.parametrize(
    "test_name, size, predict_large_called, expected_status, expected_error_msg",
    [
        # 1. Success: Normal Size (size <= 128, should use model(X[None]))
        pytest.param(
            "Success_NormalSize",
            128,
            False,  # sen2sr.predict_large should NOT be called
            200,
            None,
            id="Success_Small"
        ),
        # 2. Success: Large Size (size > 128, should use sen2sr.predict_large)
        pytest.param(
            "Success_LargeSize",
            256,
            True,  # sen2sr.predict_large SHOULD be called
            200,
            None,
            id="Success_Large"
        ),
        # 3. Failure: Core download fails (Simulated via a side effect)
        pytest.param(
            "Failure_Download",
            128,
            False,
            500,
            "Sentinel download failed",  # Assuming a custom exception from the download
            id="Failure_Download"
        ),
        # 4. Failure: Model download required but fails
        pytest.param(
            "Failure_ModelDownload",
            128,
            False,
            500,
            "Failed to download SEN2SR model",
            id="Failure_ModelDownload"
        ),
    ]
)
def test_get_sr_image_scenarios(
    monkeypatch, mock_sen2sr_dependencies, test_name, size, predict_large_called, expected_status, expected_error_msg
):
    # --- ARRANGE ---
    # Handle the Failure_Download scenario: make download_sentinel_cubo raise an error
    if test_name == "Failure_Download":
        monkeypatch.setattr(get_sr_image, "download_sentinel_cubo",
                            lambda *args, **kwargs: (_ for _ in ()).throw(Exception(expected_error_msg)))

    # Handle the Failure_ModelDownload scenario: make the model download fail
    if test_name == "Failure_ModelDownload":
        # Make os.path.exists return False, forcing the download call
        monkeypatch.setattr(os.path, "exists", lambda x: False)
        # Make mlstac.download raise an exception
        mock_sen2sr_dependencies["mlstac_download_mock"].side_effect = Exception(
            expected_error_msg)

    # --- ACT & ASSERT (Success Cases) ---
    if expected_status == 200:

        # ACT
        result_filepath = get_sr_image.get_sr_image(
            lat=LAT, lon=LON, bands=BANDS, start_date=START_DATE, end_date=END_DATE, size=size, geometry=GEOMETRY
        )
        # Get parent dir and file's basename to ignore full path assertion issues
        result_filepath = "/".join([str(result_filepath.parent).split("/").pop(), str(result_filepath.name)])
        # ASSERT
        assert str(result_filepath) == MOCK_SR_FILEPATH

        # Check conditional logic: Was predict_large called?
        if predict_large_called:
            mock_sen2sr_dependencies["predict_large_mock"].assert_called_once()
        else:
            mock_sen2sr_dependencies["predict_large_mock"].assert_not_called()

        # Check that essential functions were called
        mock_sen2sr_dependencies["crop_parcel_mock"].assert_called_once()
        mock_sen2sr_dependencies["save_to_tif_mock"].assert_called()

    # --- ACT & ASSERT (Failure Cases) ---
    elif expected_status == 500:
        with pytest.raises(Exception, match=expected_error_msg):
            get_sr_image.get_sr_image(
                lat=LAT, lon=LON, bands=BANDS, start_date=START_DATE, end_date=END_DATE, size=size
            )

        # Clean up side effects if they were set
        if test_name == "Failure_ModelDownload":
            mock_sen2sr_dependencies["mlstac_download_mock"].side_effect = None
