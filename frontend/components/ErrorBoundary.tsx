"use client";

import React, { Component } from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-[#FAFAF8] p-8">
          <div className="max-w-md w-full bg-[#FFFFFF] border border-[#E5E5E3] rounded-lg p-8 text-center shadow-lg">
            <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-[#7A1A1A]/10 flex items-center justify-center">
              <span className="text-[#7A1A1A] text-xl font-bold">!</span>
            </div>
            <h2
              className="text-xl font-bold text-[#1A1A1A] mb-2"
              style={{ fontFamily: "EB Garamond, serif" }}
            >
              Something went wrong
            </h2>
            <p className="text-sm text-[#737373] mb-6">
              {this.state.error?.message || "An unexpected error occurred."}
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-6 py-2 bg-[#C5A028] hover:bg-[#A68A1E] text-[#1A1A1A] rounded-md text-sm font-medium transition-colors"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
