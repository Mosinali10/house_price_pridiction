import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Boston Housing Analytics",
  description: "Premium data analysis platform for Boston housing prices",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen" style={{ background: "#0f172a" }}>
        {children}
      </body>
    </html>
  );
}
