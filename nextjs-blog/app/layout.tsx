import Image from "next/image";
import Link from "next/link";
import "../styles/globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const header = (
    <header>
      <div className="text-center bg-slate-800 p-8 my-6 rounded-md">
      
        <Link href="/">
          <h1 className="text-2xl text-white font-bold mt-4">Average AI Student Blog</h1>
          <br />
        </Link>
        <p className="text-slate-300"> Welcome to the Average AI Student blog 🎉<br />
        - Where I'll be taking you on a wild ride through the twisted world of Artificial Intelligence. 
        As a computer science master student.
        </p>
      </div>
    </header>
  );

  const footer = (
    <footer>
      <div className="border-t border-slate-400 mt-12 py-6 text-center text-slate-400">
        <h3>Designed by Daniel Verdugo</h3>
      </div>
    </footer>
  );

  return (
    <html>
      <head />
      <body>
        <div className="mx-auto  max-w-2xl px-6">
          {header}
          {children}
          {footer}
        </div>
      </body>
    </html>
  );
}
