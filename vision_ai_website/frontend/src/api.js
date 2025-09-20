import axios from "axios";

const API = axios.create({ baseURL: "http://localhost:8000" });

// Request interceptor to add headers if needed
API.interceptors.request.use((config) => {
  // Add any headers here if needed
  return config;
});

// Response interceptor to handle errors
API.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error("API Error:", error);
    return Promise.reject(error);
  }
);

export const detectObjects = () => API.get("/detect");
export const askAssistant = (query) => API.post("/voice", { query });
export const getNavigation = (start, end) =>
  API.post("/navigate", { start, end });

export default API;
