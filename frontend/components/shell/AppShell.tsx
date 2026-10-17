import MedicalBackdrop from "./MedicalBackdrop";
import Pulse from "./Pulse";
import Sidebar from "./Sidebar";
import ThemeToggle from "./ThemeToggle";

export default function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <>
    <MedicalBackdrop />
    <div className="shell">
      <Sidebar />
      <div className="main">
        <header className="topbar">
          <div className="topbar-start">
            <div className="badge">Research use only</div>
            <Pulse />
          </div>
          <div className="topbar-end">
            <ThemeToggle />
          </div>
        </header>
        <main className="content" id="main">
          {children}
        </main>
      </div>
    </div>
    </>
  );
}
