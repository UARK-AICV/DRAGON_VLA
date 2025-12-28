from typing import Dict, List, Optional


def build_messages(
    base64_image_url: str,
    object_names: List[str],
    valid_ids: Optional[List[int]] = None,
    reference_images: Optional[Dict[str, str]] = None,
):
    obj_list = ", ".join(object_names)
    valid_hint = ""
    if valid_ids is not None and len(valid_ids) > 0:
        # Keep it short; sort unique
        ids_str = ", ".join(str(i) for i in sorted(set(valid_ids)))
        valid_hint = f"\nValid mask numbers visible in this image are: [{ids_str}]. Use only numbers from this list."

    examples = (
        "Example A (some absent):\n"
        "Input objects: ketchup bottle , olive oil bottle, robot gripper\n"
        "Valid output:\n"
        "{ 'ketchup bottle': [], 'olive oil bottle': [3], 'robot gripper': [1] }\n\n"
        "Example B (none visible):\n"
        "Input objects: spoon, fork\n"
        "Valid output:\n"
        "{ 'spoon': [], 'fork': [] }\n"
    )

    content = [
        {
            "type": "image_url",
            "image_url": {"url": base64_image_url},
        },
    ]
    if reference_images:
        for name, url in reference_images.items():
            content.append({"type": "image_url", "image_url": {"url": url}})
            content.append(
                {
                    "type": "text",
                    "text": f"Reference crop for '{name}' from the base view.",
                }
            )
    content.append(
        {
            "type": "text",
            "text": (
                "You are given an image with segmentation masks labeled by numbers printed on the image. "
                f"Identify mask number(s) for each of: {obj_list}.\n\n"
                "Rules:\n"
                "1) Return a Python dict whose keys are exactly the given object names (same spelling), any order.\n"
                "2) Each value must be a list of INTEGER mask numbers for that object.\n"
                "3) If an object is NOT visible or you are uncertain, return []. Do not guess.\n"
                "4) A mask number must belong to at most ONE object (no duplicates across keys).\n"
                "5) An object belongs to at most ONE mask number (no duplicates across values).\n"
                "6) Do not add extra keys or text. Do not use markdown fences."
                f"{valid_hint}\n\n"
                f"{examples}"
                "Respond ONLY with a valid Python dictionary. No prose."
            ),
        }
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a careful visual assistant. "
                "Follow formatting exactly. "
                "Never return an empty dictionary. "
                "Always include every requested key."
            ),
        },
        {
            "role": "user",
            "content": content,
        },
    ]
