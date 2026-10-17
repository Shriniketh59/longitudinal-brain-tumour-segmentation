"use client";

import { useSyncExternalStore } from "react";
import { IconMoon, IconSun } from "./icons";

type Theme = "dark" | "light";

const listeners = new Set<() => void>();

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getTheme(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

// Dark reading-room theme is the default; the pre-paint script in layout applies a saved choice.
const getServerTheme = (): Theme => "dark";

export default function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, getTheme, getServerTheme);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("ns-theme", next);
    } catch {
      // Storage can be blocked; the choice then lasts for this page view only.
    }
    listeners.forEach((l) => l());
  }

  const label = theme === "dark" ? "Switch to light clinical theme" : "Switch to dark reading-room theme";

  return (
    <button type="button" className="theme-toggle" onClick={toggle} aria-label={label} title={label}>
      {theme === "dark" ? <IconSun /> : <IconMoon />}
    </button>
  );
}
