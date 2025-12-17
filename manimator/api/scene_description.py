from fastapi import HTTPException
import litellm
import os
from dotenv import load_dotenv
import ast
import json
import requests
import re
from manimator.utils.helpers import compress_pdf
from manimator.utils.system_prompts import SCENE_SYSTEM_PROMPT, IMAGE_EXTRACTION_SYSTEM_PROMPT
from manimator.few_shot.few_shot_prompts import SCENE_EXAMPLES, PDF_EXAMPLE

from serpapi import GoogleSearch

load_dotenv()


def process_prompt_scene(prompt: str) -> str:
    """Generate a scene description from a text prompt using LLM.

    This function takes a text prompt and generates a detailed scene description
    using the configured LLM model. It includes few-shot examples to improve
    the quality of generated descriptions.

    Args:
        prompt: The text prompt describing the desired scene

    Returns:
        str: Generated scene description

    Raises:
        HTTPException: If the model fails to generate a description
    """

    messages = [
        {
            "role": "system",
            "content": SCENE_SYSTEM_PROMPT,
        },
    ]
    messages.extend(SCENE_EXAMPLES)
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    response = litellm.completion(
        model=os.getenv("PROMPT_SCENE_GEN_MODEL"),
        messages=messages,
        num_retries=2,
    )
    return response.choices[0].message.content

def search_image_online(prompt: str)-> str:
    messages = [
        {
            "role": "system",
            "content": IMAGE_EXTRACTION_SYSTEM_PROMPT,
        },
    ]
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    response = litellm.completion(
        model=os.getenv("PROMPT_SCENE_GEN_MODEL"),
        messages=messages,
        num_retries=2,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        print("Output Type Not Correct")

def extract_image_files(prompt):
    prompt = json.loads(prompt)
    if len(prompt)>0:
        for i,j in enumerate(prompt):
            search_query = j["search_query"]
            params = {
            "engine": "google_images",
            "q": f"{search_query}",
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 1
            }
            search = GoogleSearch(params)
            results = search.get_dict()
            img_url = results["images_results"][0]["original"]
            file_name = img_url.split('/')[-1]
            with open(file_name, "wb") as f:
                f.write(requests.get(f"{img_url}").content)
            j["file_name"] = file_name
        
    return prompt

def process_pdf_prompt(
    file_content: bytes,
    model: str = os.getenv("PDF_SCENE_GEN_MODEL"),
    retry: bool = False,
) -> str:
    """Process a PDF file and generate a scene description using the specified model.

    Args:
        file_content: Raw PDF file bytes
        model: LLM model to use for processing. Defaults to env PDF_SCENE_GEN_MODEL
        retry: Whether this is a retry attempt and should it use the PDF_RETRY_MODEL

    Returns:
        str: Generated scene description

    Raises:
        HTTPException: If PDF processing fails or invalid input
    """
    if not file_content:
        raise HTTPException(status_code=400, detail="Empty PDF file provided")

    try:
        encoded_pdf = compress_pdf(file_content)
        messages = [
            {"role": "system", "content": SCENE_SYSTEM_PROMPT},
            *PDF_EXAMPLE,
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:application/pdf;base64,{encoded_pdf}",
                    }
                ],
            },
        ]

        response = litellm.completion(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content

    except Exception as e:
        retry_model = os.getenv("PDF_RETRY_MODEL")
        if not retry and retry_model:
            return process_pdf_prompt(file_content, model=retry_model, retry=True)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
