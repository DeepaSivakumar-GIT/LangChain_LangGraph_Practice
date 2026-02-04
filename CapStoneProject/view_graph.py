#!/usr/bin/env python3
"""
View the LangGraph workflow structure.

Usage:
  python view_graph.py
"""

from src.agent import get_agent


def main():
    agent = get_agent()
    graph = agent.get_graph()

    # Try to draw as PNG
    try:
        png_data = graph.draw_mermaid_png()
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("Graph saved to graph.png")
    except Exception as e:
        print(f"Mermaid PNG failed: {e}")

    # Print Mermaid syntax (paste into https://mermaid.live to visualize)
    print("\nMermaid diagram (paste into https://mermaid.live):\n")
    print(graph.draw_mermaid())


if __name__ == "__main__":
    main()
