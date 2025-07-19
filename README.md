# Sign Language Detection (ASL Alphabet) 🤟

**Real-time ASL alphabet recognition powered by AI**

## Table of Contents

- [Overview](#overview)
- [Inspiration](#inspiration)
- [Features](#features)
- [Demo](#demo)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Challenges & Learnings](#challenges--learnings)
- [Future Work](#future-work)
- [License](#license)

---

## Overview

This project is a web application that recognizes American Sign Language (ASL) alphabet letters in real time using your webcam and an AI model. It aims to break communication barriers and make sign language more accessible and interactive for everyone.

## Inspiration

The inspiration for this project came from a desire to bridge the communication gap for the deaf and hard-of-hearing community. By leveraging AI and computer vision, this project demonstrates how technology can make sign language more accessible, educational, and inclusive.

## Features

- 🖐️ **Real-time ASL alphabet detection** via webcam
- 🟩 **Visual boundary guide** to help users position their hand
- ⚡ **Instant feedback** with live prediction updates
- 🌐 **Modern, responsive UI** built with React
- ☁️ **Serverless AI backend** deployed on Vercel
- 🛡️ **Robust error handling** and user guidance

## Demo

**Live Demo:**  
[Frontend App](https://your-frontend-url.vercel.app/)  
[Backend API](https://sign-language-detection-backend.vercel.app/api/predict)

*(Replace with your actual deployed URLs)*

---

## How It Works

1. **User opens the web app** and allows webcam access.
2. **Webcam feed** is displayed with a green boundary box for hand positioning.
3. **User positions their hand** inside the box and clicks "Start Detection".
4. **Frames are captured** and sent to the backend API.
5. **AI model (SigLIP transformer)** predicts the ASL letter from the image.
6. **Prediction is displayed** in real time on the frontend.

---

## Tech Stack

- **Frontend:** React, Vite, react-webcam, CSS
- **Backend:** Python, Hugging Face Transformers, PyTorch, PIL
- **Model:** SigLIP (Sigmoid Loss for Language-Image Pre-training) fine-tuned for ASL alphabet
- **Deployment:** Vercel (serverless functions for backend and static hosting for frontend)

---

## Getting Started

### Prerequisites

- Node.js (for frontend)
- Python 3.9+ (for backend, if running locally)
- Vercel account (for deployment)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sign-language-detection.git
cd sign-language-detection
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### 3. Backend Setup (Local Testing)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

### 4. Deployment

- Deploy both `frontend` and `api` (backend) folders to Vercel.
- Ensure your frontend fetches predictions from `/api/predict` endpoint.

---

## Project Structure

```
Sign-Language-Detection/
  ├── frontend/         # React app (UI, webcam, API calls)
  ├── backend/          # Python backend (model, API)
  ├── api/              # (For Vercel) Serverless function for prediction
  ├── vercel.json       # Vercel configuration
```

---

## Challenges & Learnings

- **Model Optimization:** Balancing accuracy and speed for real-time inference.
- **Deployment:** Adapting AI models for serverless environments (cold starts, memory limits).
- **Data Handling:** Managing image formats and efficient data transfer between frontend and backend.
- **User Experience:** Designing an intuitive interface with clear instructions and feedback.

---

## Future Work

- Expand to full word/phrase recognition
- Support for additional sign languages
- Mobile app version
- Two-way communication (speech-to-sign)
- Improved model accuracy and speed

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

**Let’s make communication accessible for everyone!** 