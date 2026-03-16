import logging
import os
import shutil
from typing import Optional

from trailtraining import config
from trailtraining.data import combine as combine_jsons
from trailtraining.pipelines import strava as strava_pipeline
from trailtraining.providers import resolve_wellness_provider

log = logging.getLogger(__name__)


def _clean_directory(directory: str) -> None:
    if not os.path.isdir(directory):
        return
    for name in os.listdir(directory):
        p = os.path.join(directory, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except FileNotFoundError:
            pass


def _detect_provider(explicit: Optional[str] = None) -> str:
    return resolve_wellness_provider(explicit).provider


def main(
    *,
    clean: bool = False,
    clean_processing: bool = False,
    clean_prompting: bool = False,
    wellness_provider: Optional[str] = None,
) -> None:
    config.ensure_directories()

    if clean:
        clean_processing = True
        clean_prompting = True

    if clean_processing:
        _clean_directory(config.PROCESSING_DIRECTORY)
    if clean_prompting:
        _clean_directory(config.PROMPTING_DIRECTORY)

    provider = _detect_provider(wellness_provider)
    log.info("Selected wellness provider: %s", provider)

    if provider == "garmin":
        from trailtraining.pipelines import garmin as garmin_pipeline

        garmin_pipeline.main()
    else:
        from trailtraining.pipelines import intervals as intervals_pipeline

        intervals_pipeline.main()

    strava_pipeline.main()
    combine_jsons.main()

    log.info("All pipelines completed successfully.")
