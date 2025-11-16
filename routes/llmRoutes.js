import express from "express";
import { InferenceClient } from "@huggingface/inference";
import { v4 as uuidv4 } from "uuid";
import dotenv from "dotenv";
dotenv.config({
  path:"/home/shivam/Downloads/Desktop_Ubuntu/College_Project (Copy 2)/backend/.env",
});

const router = express.Router();
const HF_TOKEN = process.env.Hf_Token ;
const client = new InferenceClient(HF_TOKEN);


const conversations = new Map();

function getConversation(sessionId) {
  if (!conversations.has(sessionId)) {
    conversations.set(sessionId, []);
  }
  return conversations.get(sessionId);
}

function addMessage(sessionId, role, text) {
  const conv = getConversation(sessionId);
  const msg = { id: uuidv4(), role, text, timestamp: Date.now() };
  conv.push(msg);
  return msg;
}

function formatHistory(sessionId, systemPrompt) {
  const conv = getConversation(sessionId);
  return [
    { role: "system", content: systemPrompt },
    ...conv.map((m) => ({ role: m.role, content: m.text })),
  ];
}


router.post("/generate-image", async (req, res) => {
  try {
    console.log('Image generation request:', req.body);
    const prompt = req.body.prompt || "a futuristic city";
    
    const out = await client.textToImage({
      model: "stabilityai/stable-diffusion-xl-base-1.0",
      inputs: prompt,
    });

    const buffer = Buffer.from(await out.arrayBuffer());
    res.set("Content-Type", "image/png");
    res.send(buffer);
  } catch (err) {
    console.error('Image generation error:', err);
    res.status(500).json({ error: err.message });
  }
});


router.post("/chat", async (req, res) => {
  try {
    console.log('LLaMA chat request:', req.body);
    const { userMessage, sessionId } = req.body;
    
    if (!userMessage || !sessionId) {
      return res.status(400).json({ error: "userMessage and sessionId required" });
    }

    addMessage(sessionId, "user", userMessage);
    const history = formatHistory(sessionId, "You are a helpful assistant.");
    console.log('Chat history:', history);

    const out = await client.chatCompletion({
      model: "meta-llama/Llama-3.1-8B-Instruct",
      messages: history,
      max_tokens: 10000,
    });

    console.log('LLaMA API response:', out);
    
    const reply = out.choices[0].message.content;
    addMessage(sessionId, "assistant", reply);

    console.log('Sending reply:', reply);
    res.json({ 
      reply: reply, 
      conversationHistory: getConversation(sessionId) 
    });
  } catch (err) {
    console.error('LLaMA chat error:', err);
    res.status(500).json({ error: err.message });
  }
});

router.get("/chat/:sessionId", (req, res) => {
  const { sessionId } = req.params;
  const history = getConversation(sessionId); 
  res.json({ conversationHistory: history });
});


router.post("/qwen", async (req, res) => {
  try {
    console.log('Qwen request:', req.body);
    const { userMessage, imageUrl, sessionId } = req.body;
    
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    if (!userMessage && !imageUrl) {
      return res.status(400).json({ error: "Provide text or image" });
    }

    if (userMessage) addMessage(sessionId, "user", userMessage);
    const history = formatHistory(sessionId, "You are a helpful assistant.");

    const out = await client.chatCompletion({
      provider: "fireworks-ai",
      model: "Qwen/Qwen2.5-VL-32B-Instruct",
      messages: history,
      max_tokens: 25000,
      temperature: 1,
    });

    console.log('Qwen API response:', out);
    
    const reply = out.choices[0].message.content;
    addMessage(sessionId, "assistant", reply);

    console.log('Sending Qwen reply:', reply);
    res.json({ 
      reply: reply, 
      conversationHistory: getConversation(sessionId) 
    });
  } catch (err) {
    console.error("Qwen API Error:", err.response?.data || err.message);
    res.status(500).json({ error: err.message });
  }
});

router.get("/qwen/:sessionId", (req, res) => {
  const { sessionId } = req.params;
  const history = getConversation(sessionId);
  res.json({ conversationHistory: history });
});


router.post("/gemma", async (req, res) => {
  try {
    console.log('Gemma request:', req.body);
    const { prompt, userMessage, imageurl, sessionId } = req.body;
    
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    const message = prompt || userMessage;
    if (!message) {
      return res.status(400).json({ error: "Provide text message" });
    }

    addMessage(sessionId, "user", message);
    const history = formatHistory(sessionId, "You are a helpful AI assistant.");

    const out = await client.chatCompletion({
      model: "google/gemma-2-9b-it",
      messages: history,
      max_tokens: 5000,
      temperature: 0.7,
    });

    console.log('Gemma API response:', out);
    
    const reply = out.choices[0].message.content;
    addMessage(sessionId, "assistant", reply);

    console.log('Sending Gemma reply:', reply);
    res.json({ 
      reply: reply, 
      conversationHistory: getConversation(sessionId) 
    });
  } catch (err) {
    console.error("Gemma API Error:", err.response?.data || err.message);
    res.status(500).json({ error: `Gemma error: ${err.message}` });
  }
});

router.get("/gemma/:sessionId", (req, res) => {
  const { sessionId } = req.params;
  const history = getConversation(sessionId);
  res.json({ conversationHistory: history });
});

router.post("/deepseek", async (req, res) => {
  try {
    console.log('deepseek request:', req.body);
    const { userMessage,sessionId } = req.body;
    
    if (!sessionId) return res.status(400).json({ error: "sessionId required" });
    if (!userMessage) {
      return res.status(400).json({ error: "Provide text " });
    }

    if (userMessage) addMessage(sessionId, "user", userMessage);
    const history = formatHistory(sessionId, "You are a helpful assistant.");

    const out = await client.chatCompletion({
      provider: "sambanova",
      model: "deepseek-ai/DeepSeek-R1",
      messages: history,
      max_tokens: 25000,
      temperature: 1,
    });

    console.log('Deepseek API response:', out);
    
    const reply = out.choices[0].message.content;
    addMessage(sessionId, "assistant", reply);

    console.log('Sending Deepseek reply:', reply);
    res.json({ 
      reply: reply, 
      conversationHistory: getConversation(sessionId) 
    });
  } catch (err) {
    console.error("Deepseek API Error:", err.response?.data || err.message);
    res.status(500).json({ error: err.message });
  }
});

router.get("/deepseek/:sessionId", (req, res) => {
  const { sessionId } = req.params;
  const history = getConversation(sessionId);
  res.json({ conversationHistory: history });
});

router.post("/Chrono-Edit", async (req, res) => {
  try {
    console.log('Chrono-Edit request received');
    const { userMessage, imageData, imageUrl } = req.body;
    if (!imageData && !imageUrl) {
      return res.status(400).json({ error: "Image is required for Chrono-Edit" });
    }

    const prompt = userMessage || "Edit this image";
    console.log('Processing image with prompt:', prompt);
    
    // Handle base64 image data - convert to Buffer, then to Blob (which has arrayBuffer method)
    let imageInput = null;
    if (imageData) {
      // Remove data URL prefix if present (e.g., "data:image/png;base64,")
      const base64Data = imageData.includes(',') 
        ? imageData.split(',')[1] 
        : imageData;
      try {
        // Convert base64 to Buffer first
        const imageBuffer = Buffer.from(base64Data, 'base64');
        // Create Blob from Buffer - Blob has arrayBuffer() method that API expects
        // Node.js 18+ supports Blob natively
        imageInput = new Blob([imageBuffer], { type: 'image/png' });
        console.log('Image Blob created from base64, size:', imageBuffer.length);
      } catch (err) {
        return res.status(400).json({ error: "Invalid base64 image data" });
      }
    } else if (imageUrl && imageUrl.startsWith('data:')) {
      // If imageUrl is provided as base64 data URL, handle it
      const base64Data = imageUrl.split(',')[1];
      try {
        const imageBuffer = Buffer.from(base64Data, 'base64');
        imageInput = new Blob([imageBuffer], { type: 'image/png' });
        console.log('Image Blob created from base64 URL, size:', imageBuffer.length);
      } catch (err) {
        return res.status(400).json({ error: "Invalid base64 image URL" });
      }
    }

    if (!imageInput) {
      return res.status(400).json({ error: "Invalid image data" });
    }

    console.log('Calling ChronoEdit API with fal-ai provider...');
    
    // Use imageToImage with fal-ai provider
    // fal-ai provider expects an object with arrayBuffer() method (like Blob)
    const image = await client.imageToImage({
      provider: "fal-ai",
      model: "nvidia/ChronoEdit-14B-Diffusers",
      inputs: imageInput,
      parameters: {
        prompt: prompt,
      },
    });

    console.log('ChronoEdit API response received');
    
    // Convert Blob to buffer and send as PNG
    const buffer = Buffer.from(await image.arrayBuffer());
    res.set("Content-Type", "image/png");
    res.send(buffer);
  } catch (err) {
    console.error("Chrono-Edit Error:", {
      message: err.message,
      stack: err.stack
    });
    res.status(500).json({ 
      error: err.message || "Failed to process image editing"
    });
  }
});



export default router;