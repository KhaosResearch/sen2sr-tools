import argparse
import structlog
import sys
from pathlib import Path
from typing import List

logger = structlog.get_logger()

class VerboseArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise RuntimeError(f"Argument error: {message}")


# --- Helper to load dynamic logic and run command ---

def run_command(command_name: str, args: argparse.Namespace):
    """Dynamically imports and runs the function corresponding to the command."""
    # Lazy import to speed up initial CLI startup
    try:
        from sen2sr_tools.get_sr_image import (
            get_sr_image,
            download_sentinel_cubo,
            crop_png_from_tif,
        )
    except ImportError:
        logger.error(f"Could not import logic from sen2sr_tools.get-sr-image. Ensure the package is installed in editable mode.")
        sys.exit(1)

    # Dispatch logic based on command
    if command_name == "get-sr-image":
        logger.info("Starting Super Resolution task", 
                    latitude=args.latitude, longitude=args.longitude, 
                    start_date=args.start_date, end_date=args.end_date,
                    geojson_path=args.geojson_path
        )
        result = get_sr_image(
            lat=args.latitude, 
            lon=args.longitude, 
            bands=args.bands, 
            start_date=args.start_date, 
            end_date=args.end_date, 
            size=args.size,
            geojson_path=args.geojson_path
        )
        logger.info(f"✅ Get SR Image task completed. Result (filepath):\n{result}")
        return result

    elif command_name == "download-cubo":
        logger.info("Starting Sentinel-2 Cubo download", start_date=args.start_date, end_date=args.end_date)
        
        # Note: The download function returns two values, we'll only print the path/date for CLI feedback
        data, date = download_sentinel_cubo(
            lat=args.latitude, 
            lon=args.longitude, 
            bands=args.bands, 
            start_date=args.start_date, 
            end_date=args.end_date, 
            size=args.size,
            crs=args.crs,
            cloud_threshold=args.cloud_threshold
        )
        logger.info(f"✅ Download successful for acquisition date: {date}")
        return data, date

    elif command_name == "crop-polygon":
        logger.info("Starting parcel cropping.", tif_path=args.raster_path)
        
        # Note: The crop function needs a `date` string as the second argument,
        # which it uses for naming the output file. We'll add a required argument for this.
        if not args.acquisition_date:
             raise ValueError("The --acquisition-date argument is required for file naming.")
             
        out_path = crop_png_from_tif(
            raster_path=args.raster_path, 
            geojson_path=args.geojson_path, 
            date=args.acquisition_date
        )
        logger.info(f"✅ Cropping successful. Output path: {out_path}")
        return out_path
    
# --- Argument Parser Setup ---

def get_parser():
    parser = VerboseArgumentParser(description="SEN2SR Tools: Command Line Interface for Sentinel-2 Super Resolution.")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    subparsers.required = True

    # =========================================================================
    # 1. get-sr-image (Main Function)
    # =========================================================================
    get_sr_image_parser = subparsers.add_parser(
        "get-sr-image", help="Get SR image from downloaded Sentinel's imagery data and load up SEN2SR model to Super-Resolve it."
    )
    get_sr_image_parser.add_argument(
        "--latitude", type=float, help="Latitude coordinate component (e.g., 37.265840).", required=True, metavar="FLOAT",
    )
    get_sr_image_parser.add_argument(
        "--longitude", type=float, help="Longitude coordinate component (e.g., -4.593406).", required=True, metavar="FLOAT",
    )
    get_sr_image_parser.add_argument(
        "--start-date", type=str, help="Initial date in search range in ISO format (yyyy-mm-dd).", required=True, metavar="STR",
    )
    get_sr_image_parser.add_argument(
        "--end-date", type=str, help="Final date in search range in ISO format (yyyy-mm-dd).", required=True, metavar="STR",
    )
    get_sr_image_parser.add_argument(
        "--bands", nargs='+', default=["B08", "B02", "B03", "B04", "SCL"], 
        help="Image bands to retrieve (space-separated, e.g., B08 B02 B03 B04 SCL).", required=False,
    )
    get_sr_image_parser.add_argument(
        "--size", type=int, default=128, help="Size of the image tile in pixels (e.g., 128 for 128x128).", required=False, metavar="INT",
    )
    get_sr_image_parser.add_argument(
        "--geojson-path", type=str, default="", help="GeoJSON to crop SR image filepath. If none is provided, it downloads the image in sizexsize", metavar="STR",
    )


    # =========================================================================
    # 2. download_cubo (Helper Function)
    # =========================================================================
    download_cubo_parser = subparsers.add_parser(
        "download-cubo", help="Download raw Sentinel-2 imagery data using Cubo and filter for cloudless data."
    )
    # Re-use common arguments
    download_cubo_parser.add_argument(
        "--latitude", type=float, help="Latitude coordinate component.", required=True, metavar="FLOAT",
    )
    download_cubo_parser.add_argument(
        "--longitude", type=float, help="Longitude coordinate component.", required=True, metavar="FLOAT",
    )
    download_cubo_parser.add_argument(
        "--start-date", type=str, help="Initial date in search range (yyyy-mm-dd).", required=True, metavar="STR",
    )
    download_cubo_parser.add_argument(
        "--end-date", type=str, help="Final date in search range (yyyy-mm-dd).", required=True, metavar="STR",
    )
    download_cubo_parser.add_argument(
        "--bands", nargs='+', default=["B08", "B02", "B03", "B04", "SCL"], help="Image bands to retrieve.",
    )
    download_cubo_parser.add_argument(
        "--size", type=int, default=128, help="Size of the image tile in pixels.", metavar="INT",
    )
    # Specific arguments for download
    download_cubo_parser.add_argument(
        "--crs", type=str, default=None, help="Optional: Coordinate Reference System (e.g., EPSG:32630). Calculated from Lon/Lat if not provided.", metavar="STR",
    )
    download_cubo_parser.add_argument(
        "--cloud-threshold", type=float, default=0.01, help="Max percentage of cloud density tolerated (0.0 to 1.0).", metavar="FLOAT",
    )
    download_cubo_parser.add_argument(
        "--geojson-path", type=str, default="", help="GeoJSON to crop SR image filepath. If none is provided, it downloads the image in sizexsize", metavar="STR",
    )


    # =========================================================================
    # 3. crop_parcel (Helper Function)
    # =========================================================================
    crop_parcel_parser = subparsers.add_parser(
        "crop-polygon", help="Crops a parcel from a Super-Resolved TIF using a local GeoJSON file."
    )
    crop_parcel_parser.add_argument(
        "--raster-path", type=Path, help="Path to the uncropped SR TIF image.", required=True, metavar="PATH",
    )

    crop_parcel_parser.add_argument(
        "--geojson-path", type=Path, help="Path to the GeoJSON file to crop the SR TIF image.", required=True, metavar="PATH",
    )

    crop_parcel_parser.add_argument(
        "--acquisition-date", type=str, help="Acquisition date (yyyy-mm-dd) used to name the output file.", required=True, metavar="STR",
    )
    
    return parser

def main():
    args = get_parser().parse_args() 
    
    # Run the command
    result = run_command(args.command, args)

    logger.info(f"RESULT:")
    logger.info(f"{result}")

if __name__ == "__main__":
    main()
