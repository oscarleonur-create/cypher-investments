/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0e14",
        panel: "#121722",
        "panel-2": "#1a2030",
        border: "#232b3b",
        muted: "#8b97ad",
        text: "#e6eaf2",
        pos: "#19c37d",
        neg: "#ef4444",
        warn: "#f59e0b",
        accent: "#3b82f6",
      },
    },
  },
  plugins: [],
};
