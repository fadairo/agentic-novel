# Agentic Novel Writing with ChatGPT 01 & Pinecone

This project demonstrates an **agentic**, **scene-by-scene** approach to writing a novel using:
- **ChatGPT 01** for text generation and literary critiques
- **Pinecone** for optional context storage and retrieval

The system:
1. **Generates** a draft scene based on a provided outline.  
2. **Critiques** each scene with a "literary agent."  
3. **Refines** the draft according to critique.  
4. **Combines** refined scenes into a chapter, then **critiques and refines** the entire chapter.  
5. **Saves** each final chapter locally as a `.txt` file.  

You can optionally **store** and **query** scene data or user clarifications in Pinecone to build a more robust memory system during novel creation.

---

## Table of Contents
1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Setup](#setup)  
6. [Usage](#usage)  
7. [Configuration Details](#configuration-details)  
8. [Extending the Project](#extending-the-project)  
9. [License](#license)

---

## Features

- **Snowflake-Inspired Method**: Write a novel in iterative stages (scene by scene, then refine the chapter).  
- **Literary Critique**: A “LiteraryAgent” class critiques each scene or chapter for style, pacing, character consistency, etc.  
- **Optional Pinecone Context Storage**: Store or retrieve outlines, drafts, critiques, or user clarifications from Pinecone for improved context.  
- **Local Chapter Saving**: Each final version of a chapter is saved to a user-specified folder.  

---

## Project Structure

