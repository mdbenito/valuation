import logging
import os
from pathlib import Path

import mkdocs.plugins

logger = logging.getLogger(__name__)

root_dir = Path(__file__).parent.parent
docs_dir = root_dir / "docs"
contributing_file = root_dir / "CONTRIBUTING.md"
target_filepath = docs_dir / "development_guidelines.md"


@mkdocs.plugins.event_priority(100)
def on_pre_build(config):
    logger.info("Temporarily copying contributing guide to docs directory")
    try:
        if os.path.getmtime(contributing_file) <= os.path.getmtime(target_filepath):
            logger.info(
                f"Contributing guide '{os.fspath(contributing_file)}' hasn't been updated, skipping."
            )
            return
    except FileNotFoundError:
        pass
    logger.info(
        f"Creating symbolic link for '{os.fspath(contributing_file)}' "
        f"at '{os.fspath(target_filepath)}'"
    )
    target_filepath.symlink_to(contributing_file)

    logger.info("Finished copying contributing guide to docs directory")


@mkdocs.plugins.event_priority(-100)
def on_shutdown():
    logger.info("Removing temporary contributing guide in docs directory")
    target_filepath.unlink()
