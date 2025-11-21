import os
import shutil
import wandb
import weave
import json
from datetime import datetime
import logging
import glob




# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_wandb_run(
    project=os.environ.get("WANDB_PROJECT_NAME"),
    entity=os.environ.get("WANDB_ENTITY"),
    resume="allow",
    id=wandb.util.generate_id(),
    stage="planning",
):
    # Main execution
    # Initialize W&B run (assuming API connection is active)

    wandb.init(project=project, entity=entity, resume=resume, id=id,  settings=wandb.Settings(init_timeout=300))
    wandb.config.update({"stage": stage}, allow_val_change=True)  # Set stage at the start

    # Initialize Weave project - read project name from environment with a sensible default
    weave.init(project_name=os.environ.get("WEAVE_PROJECT_NAME"))

    logger.info(f"W&B run initialized with ID: {wandb.run.id}")
    return wandb.run.id

def _prepare_artifact_metadata_and_aliases(output_dir, artifact_type, step, is_resumed, stage, phase):
    timestamp = datetime.now().isoformat()
    metadata = {
        "stage": stage or "unknown",
        "phase": phase or "unknown",
        "source_dir": output_dir,
        "timestamp": timestamp,
        "step": step,
        "is_resumed": is_resumed,
        "run_id": wandb.run.id if wandb.run else "unknown",
    }

    artifact_name = (
        f"paper2code-{stage}-{phase}"
        if stage and phase
        else "paper2code-artifacts"
    )
    aliases = []
    if step is not None:
        aliases.append(f"step-{step}")
    if is_resumed:
        aliases.append("resumed")
    aliases.append("latest")

    return artifact_name, metadata, aliases

def upload_zipped_artifact(
    output_dir,
    file_pattern,
    artifact_type="dataset",
    step=None,
    is_resumed=False,
    stage=None,
    phase=None,
):
    """
    Upload JSON artifacts to W&B as a zipped archive, preserving metadata.
    """
    if not wandb.run:
        logger.warning("W&B run not initialized. Skipping artifact upload.")
        return

    artifact_files = glob.glob(os.path.join(output_dir, f"*{file_pattern}"))
    if not artifact_files:
        logger.warning(f"No files found matching '{file_pattern}' in {output_dir}. Skipping upload.")
        return

    try:
        temp_dir = os.path.join(output_dir, "temp_artifacts")
        os.makedirs(temp_dir, exist_ok=True)

        for file_path in artifact_files:
            shutil.copy(file_path, os.path.join(temp_dir, os.path.basename(file_path)))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_name = f"artifacts_step_{step}_{timestamp}.zip" if step else f"artifacts_{timestamp}.zip"
        zip_path = os.path.join(output_dir, zip_name)
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", temp_dir)
        shutil.rmtree(temp_dir)

        artifact_name, metadata, aliases = _prepare_artifact_metadata_and_aliases(
            output_dir, artifact_type, step, is_resumed, stage, phase
        )

        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        artifact.add_file(zip_path, name=zip_name)

        logger.info(f"Logging zipped artifact '{artifact_name}' to W&B with aliases {aliases}...")
        wandb.log_artifact(artifact, aliases=aliases)
        logger.info(f"Artifact '{artifact_name}' logged successfully.")

        if phase:
            wandb.log({"phase": phase})
        os.remove(zip_path)

    except Exception as e:
        logger.warning(f"Failed zipped artifact upload for {output_dir}: {e}")
        import traceback
        traceback.print_exc()

def upload_unzipped_artifact(
    output_dir,
    file_pattern,
    artifact_type="dataset",
    step=None,
    is_resumed=False,
    stage=None,
    phase=None,
):
    """
    Upload JSON artifacts to W&B without zipping, preserving metadata.
    """
    if not wandb.run:
        logger.warning("W&B run not initialized. Skipping artifact upload.")
        return

    artifact_files = glob.glob(os.path.join(output_dir, f"*{file_pattern}"))
    if not artifact_files:
        logger.warning(f"No files found matching '{file_pattern}' in {output_dir}. Skipping upload.")
        return

    try:
        artifact_name, metadata, aliases = _prepare_artifact_metadata_and_aliases(
            output_dir, artifact_type, step, is_resumed, stage, phase
        )

        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        for file_path in artifact_files:
            artifact.add_file(file_path, name=os.path.basename(file_path))

        logger.info(f"Logging unzipped artifact '{artifact_name}' to W&B with aliases {aliases}...")
        wandb.log_artifact(artifact, aliases=aliases)
        logger.info(f"Artifact '{artifact_name}' logged successfully.")

        if phase:
            wandb.log({"phase": phase})

    except Exception as e:
        logger.warning(f"Failed unzipped artifact upload for {output_dir}: {e}")
        import traceback
        traceback.print_exc()


def upload_artifact(
    output_dir,
    file_pattern,
    artifact_type="dataset",
    step=None,
    is_resumed=False,
    stage=None,
    phase=None,
    use_zip=False,
):
    """
    Upload JSON artifacts to W&B, either zipped or unzipped, with fixed metadata.
    """
    if use_zip:
        upload_zipped_artifact(
            output_dir=output_dir,
            file_pattern=file_pattern,
            artifact_type=artifact_type,
            step=step,
            is_resumed=is_resumed,
            stage=stage,
            phase=phase,
        )
    else:
        upload_unzipped_artifact(
            output_dir=output_dir,
            file_pattern=file_pattern,
            artifact_type=artifact_type,
            step=step,
            is_resumed=is_resumed,
            stage=stage,
            phase=phase,
        )


def log_stage(stage):
    """
    Update W&B config with the current stage, allowing value changes.
    """
    wandb.config.update({"stage": stage}, allow_val_change=True)
    logger.info(f"Updated W&B config with stage: {stage}")


def log_phase(phase):
    """
    Log the completed phase to W&B metrics.
    """
    wandb.log({"phase": phase})
    logger.info(f"Logged phase: {phase}")

def generic_wandb_logger(
    output_dir,
    stage,
    phase=None,
    step=None,
    is_resumed=False,
    artifact_type="dataset",
    file_pattern="*.json",
):
    """
    Generic W&B logger for artifact upload and phase logging.

    Args:
        output_dir (str): Directory containing JSON files.
        stage (str): Current stage (e.g., "planning", "analyzing", "coding").
        phase (str, optional): Current phase (e.g., "initialization", "api_call", "save").
        step (int, optional): Current processing step.
        is_resumed (bool): Whether this run resumed from a prior state.
        artifact_type (str): W&B artifact type.
        file_pattern (str): Glob pattern for files to upload.
    """
    upload_artifact(
        output_dir, file_pattern, artifact_type, step, is_resumed, stage, phase, use_zip=False
    )
    if phase:
        log_phase(phase)



    