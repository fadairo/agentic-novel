import os
import json
import time
from typing import Dict, Any, List, Optional

import pinecone  # pip install pinecone-client
# If you're using OpenAI for embeddings:
# import openai

# ----------------------------------------------------------------------
# 1. ChatGPT 01 Interface
#    This is a placeholder for however you are calling ChatGPT 01
#    (e.g., via OpenAI’s ChatCompletion or your custom API).
# ----------------------------------------------------------------------

class ChatGPT01LLM:
    """
    A placeholder class that wraps calls to ChatGPT 01.
    Adapt this to your actual ChatGPT 01 calling mechanism.
    """
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        # e.g., openai.api_key = self.api_key

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Example of calling ChatGPT 01. You'd likely use something like:
            openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        This is just a stub returning mock text.
        """
        # Example Pseudocode:
        # response = openai.ChatCompletion.create(
        #     model="gpt-4",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=max_tokens,
        #     temperature=0.7
        # )
        # return response.choices[0].message.content
        return f"[ChatGPT 01 Mock Output for prompt: {prompt[:60]} ...]"


# ----------------------------------------------------------------------
# 2. (Optional) Pinecone Integration
#    We might store user clarifications or scene embeddings here.
# ----------------------------------------------------------------------

class PineconeContextStore:
    """
    Demonstrates how to store and retrieve context in Pinecone.
    This can be used for retrieval-augmented generation, references, etc.
    """
    def __init__(self, api_key: str, environment: str, index_name: str):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        
        pinecone.init(api_key=api_key, environment=environment)
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(name=self.index_name, dimension=1536) 
            # dimension depends on your embedding model
        
        self.index = pinecone.Index(name=self.index_name)

    def embed_text(self, text: str) -> List[float]:
        """
        Convert text to embedding vector. 
        You’d likely use OpenAI or another service to get a 1536-dim vector:
            
            embedding = openai.Embedding.create(input=[text], model="text-embedding-ada-002")
            vector = embedding["data"][0]["embedding"]
        
        Here we just return a mock vector.
        """
        return [0.0] * 1536  # placeholder

    def store_context(self, text: str, metadata: Dict[str, Any], namespace: str = "default"):
        """
        Stores the text (as an embedding) in Pinecone.
        """
        embedding_vector = self.embed_text(text)
        unique_id = metadata.get("id", f"item-{time.time()}")
        self.index.upsert(
            vectors=[(unique_id, embedding_vector, metadata)],
            namespace=namespace
        )

    def query_context(self, query_text: str, top_k: int = 3, namespace: str = "default") -> List[Dict[str, Any]]:
        """
        Queries Pinecone for the most relevant context to `query_text`.
        Returns metadata for top_k results.
        """
        query_vector = self.embed_text(query_text)
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        return results["matches"]


# ----------------------------------------------------------------------
# 3. Literary Agent (Critique) Class
# ----------------------------------------------------------------------

class LiteraryAgent:
    """
    Acts as a “literary agent” that critiques each scene or chapter using ChatGPT 01.
    """
    def __init__(self, critique_llm: ChatGPT01LLM):
        self.critique_llm = critique_llm

    def critique_text(self, text: str, context: Optional[str] = None) -> str:
        """
        Uses ChatGPT 01 to provide feedback on the scene/chapter text.
        """
        critique_prompt = (
            f"You are a literary agent. Assess the following text for style, clarity, "
            f"pacing, character consistency, and overall quality.\n\n"
            f"Context:\n{context}\n\n"
            f"Text to critique:\n{text}\n\n"
            f"Provide a concise critique and suggestions for improvement."
        )
        return self.critique_llm.generate(prompt=critique_prompt, max_tokens=512)


# ----------------------------------------------------------------------
# 4. Scene-by-Scene Writing with Snowflake Method
# ----------------------------------------------------------------------

def write_scene(
    writing_llm: ChatGPT01LLM,
    literary_agent: LiteraryAgent,
    scene_outline: str,
    overall_context: str,
    user_clarifications: Dict[str, Any] = None,
    pinecone_store: Optional[PineconeContextStore] = None
) -> str:
    """
    Composes a scene based on the outline, critiques it, and refines if necessary.
    Optionally stores or retrieves context from Pinecone.
    """
    # (Optional) retrieve relevant user clarifications from Pinecone, etc.
    # For demonstration, we just incorporate user_clarifications directly.

    # Step 1: Generate a draft
    prompt = (
        f"You are an expert novelist using the Snowflake Method. "
        f"Write a scene based on this outline:\n\n{scene_outline}\n\n"
        f"Overall context:\n{overall_context}\n\n"
        f"User clarifications: {json.dumps(user_clarifications, indent=2)}\n\n"
        f"Use an engaging, consistent tone. Aim for about 800-1500 words."
    )
    draft = writing_llm.generate(prompt=prompt, max_tokens=1500)

    # Step 2: Literary agent critique
    critique = literary_agent.critique_text(text=draft, context=overall_context)
    print("\n--- Literary Agent Critique for This Scene ---")
    print(critique)
    print("---------------------------------------------\n")

    # (Optional) store the draft or clarifications in Pinecone
    if pinecone_store:
        pinecone_store.store_context(
            text=draft,
            metadata={
                "id": f"scene-{time.time()}",
                "type": "draft",
                "outline": scene_outline,
                "critique": critique
            },
            namespace="novel"
        )

    # Step 3: Refine the draft with the critique
    refine_prompt = (
        f"You wrote the following scene:\n\n{draft}\n\n"
        f"Here is the critique:\n{critique}\n\n"
        f"Refine the scene according to the critique while preserving style and storyline. "
        f"Please incorporate suggestions carefully."
    )
    refined_draft = writing_llm.generate(prompt=refine_prompt, max_tokens=1500)

    return refined_draft


# ----------------------------------------------------------------------
# 5. Chapter Writing Function
# ----------------------------------------------------------------------

def write_chapter(
    writing_llm: ChatGPT01LLM,
    literary_agent: LiteraryAgent,
    chapter_outline: List[str],
    overall_context: str,
    chapter_number: int,
    user_clarifications: Dict[str, Any] = None,
    save_directory: str = "./chapters",
    pinecone_store: Optional[PineconeContextStore] = None
) -> str:
    """
    Writes a chapter scene by scene, then critiques the full chapter output.
    """
    print(f"=== Writing Chapter {chapter_number} ===")
    chapter_scenes_text = []
    
    for i, scene_outline in enumerate(chapter_outline, start=1):
        print(f"--- Scene {i} of Chapter {chapter_number} ---")
        
        # If needed, you could ask user for clarifications or retrieve from Pinecone
        scene_text = write_scene(
            writing_llm=writing_llm,
            literary_agent=literary_agent,
            scene_outline=scene_outline,
            overall_context=overall_context,
            user_clarifications=user_clarifications,
            pinecone_store=pinecone_store
        )
        
        chapter_scenes_text.append(scene_text)
        time.sleep(1)  # simulating a delay, if desired

    # Combine scenes into the full chapter text
    chapter_text = "\n\n".join(chapter_scenes_text)

    # Critique the full chapter
    chapter_critique = literary_agent.critique_text(text=chapter_text, context=overall_context)
    print(f"=== Full Chapter {chapter_number} Critique ===\n{chapter_critique}\n")

    # Optionally refine the entire chapter after critique
    refine_chapter_prompt = (
        f"You wrote the following chapter:\n\n{chapter_text}\n\n"
        f"Here is the critique:\n{chapter_critique}\n\n"
        f"Refine the chapter according to the critique while preserving continuity, "
        f"plot logic, and character arcs."
    )
    refined_chapter_text = writing_llm.generate(prompt=refine_chapter_prompt, max_tokens=3000)

    # Save the final refined chapter locally
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    chapter_path = os.path.join(save_directory, f"chapter_{chapter_number}.txt")
    with open(chapter_path, "w", encoding="utf-8") as f:
        f.write(refined_chapter_text)

    print(f"Chapter {chapter_number} saved to: {chapter_path}")
    return refined_chapter_text


# ----------------------------------------------------------------------
# 6. Main Novel Writing Flow
# ----------------------------------------------------------------------

def write_novel(
    chatgpt_api_key: str,
    pinecone_api_key: str,
    pinecone_environment: str,
    pinecone_index_name: str,
    novel_title: str = "Untitled Novel",
    genre: str = "Fantasy",
    setting: str = "Medieval Kingdom",
    character_list: List[str] = None,
    outline: Dict[int, List[str]] = None,
    user_clarifications: Dict[str, Any] = None,
    chapters_to_write: Optional[List[int]] = None,
    save_directory: str = "./chapters",
    use_pinecone: bool = False
) -> None:
    """
    High-level function that coordinates writing an entire novel
    by iterating over the provided outline, chapter by chapter.
    """
    # Initialize the ChatGPT 01 LLM
    chatgpt_llm = ChatGPT01LLM(api_key=chatgpt_api_key)

    # Optional Pinecone context store
    pinecone_store = None
    if use_pinecone:
        pinecone_store = PineconeContextStore(
            api_key=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name
        )

    # Create a LiteraryAgent for critiques
    literary_agent = LiteraryAgent(critique_llm=chatgpt_llm)

    # Summarize high-level context
    overall_context = (
        f"Novel Title: {novel_title}\n"
        f"Genre: {genre}\n"
        f"Setting: {setting}\n"
        f"Characters: {', '.join(character_list or [])}\n"
        f"Overall Outline: {json.dumps(outline, indent=2)}\n"
        f"User Clarifications: {json.dumps(user_clarifications or {}, indent=2)}\n"
    )

    # Default to writing every chapter if not specified
    if not chapters_to_write:
        chapters_to_write = sorted(outline.keys())

    # Iterate through each chapter in the outline
    for chapter_number in chapters_to_write:
        chapter_outline = outline[chapter_number]

        print(f"\n=========================================================")
        print(f"Starting Chapter {chapter_number}")
        print(f"=========================================================")
        
        # Write the chapter (scene by scene)
        final_chapter_text = write_chapter(
            writing_llm=chatgpt_llm,
            literary_agent=literary_agent,
            chapter_outline=chapter_outline,
            overall_context=overall_context,
            chapter_number=chapter_number,
            user_clarifications=user_clarifications,
            save_directory=save_directory,
            pinecone_store=pinecone_store
        )
        
        # Optionally, confirm from user if we should proceed or do more edits
        # user_input = input("Type 'continue' to proceed to the next chapter or 'edit' to refine again: ")
        # if user_input.lower() == 'edit':
        #     # possibly do more refinement steps here
        #     pass
        
        time.sleep(1)

    print("\n>>> Novel writing complete! All selected chapters have been drafted and saved. <<<")


# ----------------------------------------------------------------------
# 7. Example Usage (if you were to run this script directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # You could load these from environment variables or a config file
    CHATGPT_API_KEY = "YOUR_CHATGPT_API_KEY"
    PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
    PINECONE_ENVIRONMENT = "YOUR_PINECONE_ENVIRONMENT"
    PINECONE_INDEX_NAME = "novel-context"

    # Mock outline for demonstration
    example_outline = {
        1: [
            "Scene 1.1: Introduce protagonist in a normal day routine.",
            "Scene 1.2: An unexpected letter arrives, creating upheaval."
        ],
        2: [
            "Scene 2.1: Protagonist travels to meet an old friend for advice.",
            "Scene 2.2: A confrontation reveals deeper threats."
        ],
        # Additional chapters...
    }
    example_characters = ["Alice", "Bob the Mentor", "Villain X"]

    # Example clarifications
    user_clarifications = {
        "preferred_writing_style": "Third-person limited, comedic undertones",
        "desired_chapter_length": "around 3000 words each"
    }

    # Run the novel writing (for demonstration, we only do Chapter 1)
    write_novel(
        chatgpt_api_key=CHATGPT_API_KEY,
        pinecone_api_key=PINECONE_API_KEY,
        pinecone_environment=PINECONE_ENVIRONMENT,
        pinecone_index_name=PINECONE_INDEX_NAME,
        novel_title="The Clockwork Galleon",
        genre="Steampunk Fantasy",
        setting="A floating archipelago city",
        character_list=example_characters,
        outline=example_outline,
        user_clarifications=user_clarifications,
        chapters_to_write=[1],
        save_directory="./my_novel_chapters",
        use_pinecone=True  # set to False if you don't want to use Pinecone
    )
