import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#111a1f",
        paper: "#f4ead9",
        bone: "#fff8ea",
        moss: "#28695d",
        coral: "#c65f46",
        amber: "#c99634",
        sky: "#2e7690",
        signal: "#72a6a4",
        night: "#172329",
        slateblue: "#3f5065"
      },
      fontFamily: {
        display: ["Bahnschrift", "Aptos Display", "Segoe UI Variable Display", "sans-serif"],
        body: ["Aptos", "Segoe UI Variable Text", "Trebuchet MS", "sans-serif"],
        mono: ["Cascadia Mono", "Consolas", "monospace"]
      },
      boxShadow: {
        soft: "0 18px 55px rgba(17,26,31,0.14)",
        instrument: "0 24px 80px rgba(17,26,31,0.20)"
      },
      backgroundImage: {
        grid: "linear-gradient(rgba(17,26,31,0.055) 1px, transparent 1px), linear-gradient(90deg, rgba(17,26,31,0.055) 1px, transparent 1px)"
      }
    }
  },
  plugins: []
};

export default config;
