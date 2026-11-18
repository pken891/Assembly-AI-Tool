import sys
from datetime import datetime
from openai import OpenAI
from ddgs import DDGS
import trafilatura
from rich.console import Console
from rich.panel import Panel

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="sk-no-key-required"
)

console = Console()

def get_system_prompt() -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    return (
        f"You are a direct, technically accurate coding and general knowledge assistant. "
        f"Today's date is {today}. "
        f"For coding questions, write clean, bug-free code. For general knowledge or sports/news "
        f"questions, answer concisely in plain prose—do NOT include code examples unless asked. "
        f"When LIVE WEB SEARCH RESULTS are provided, base your answer strictly on those facts "
        f"and cite the URLs."
    )

def generate_search_keywords(user_prompt: str) -> str:
    """Uses local Qwen3 to turn a conversational prompt into a clean 4-6 word search query."""
    today_str = datetime.now().strftime("%B %Y")
    try:
        resp = client.chat.completions.create(
            model="qwen3-coder",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Today is {today_str}. Convert the user's message into a concise 3 to 6 word "
                        f"web search query. Strip out conversational filler like 'testing web search'. "
                        f"Do NOT use punctuation or quotes. Output ONLY the search keywords."
                    )
                },
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=25
        )
        keywords = resp.choices[0].message.content.strip().strip('"\'')
        return keywords if keywords else user_prompt
    except Exception:
        return user_prompt

def fetch_web_context(raw_query: str) -> str:
    """Fetches both Web and News results in isolated try-blocks so one never crashes the other."""
    query = generate_search_keywords(raw_query)
    console.print(f"[bold cyan]Optimized Search Query:[/bold cyan] '{query}'")

    combined_results = []
    ddgs = DDGS()

    # 1. Isolated try-block for Standard Web Search (most reliable)
    try:
        web_hits = ddgs.text(query, max_results=3) or []
        for w in web_hits:
            combined_results.append({
                "type": "WEB",
                "title": w.get("title", "Untitled"),
                "url": w.get("href", ""),
                "date": "Web Result",
                "snippet": w.get("body", "")
            })
    except Exception as e:
        console.print(f"  [dim yellow]Web search notice: {e}[/dim yellow]")

    # 2. Isolated try-block for News Search (won't kill Web results if it finds nothing)
    try:
        news_hits = ddgs.news(query, max_results=3) or []
        for n in news_hits:
            combined_results.append({
                "type": "NEWS",
                "title": n.get("title", "Untitled"),
                "url": n.get("url", ""),
                "date": n.get("date", "Recent"),
                "snippet": n.get("body", "")
            })
    except Exception:
        pass

    if not combined_results:
        console.print("[bold red]Both Web and News searches returned 0 results.[/bold red]")
        return ""

    # Deduplicate URLs and keep up to 4 results
    seen_urls = set()
    filtered_results = []
    for item in combined_results:
        if item["url"] and item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            filtered_results.append(item)
        if len(filtered_results) == 4:
            break

    context_blocks = []
    for idx, res in enumerate(filtered_results, 1):
        title = res["title"]
        url = res["url"]
        snippet = res["snippet"]
        rtype = res["type"]
        rdate = res["date"]
        console.print(f"  [dim][{idx}] [{rtype}] Scraping:[/dim] {title} ([underline]{url}[/underline])")

        page_text = ""
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                extracted = trafilatura.extract(
                    downloaded,
                    include_tables=True,
                    include_comments=False
                )
                if extracted:
                    page_text = extracted[:2500]
        except Exception:
            pass

        block = (
            f"--- SOURCE {idx} [{rtype} | Date: {rdate}]: {title} ---\n"
            f"URL: {url}\n"
            f"SEARCH SUMMARY SNIPPET: {snippet}\n"
        )
        if page_text:
            block += f"EXTRACTED PAGE TEXT:\n{page_text}\n"
        context_blocks.append(block)

    return "\n".join(context_blocks)

def main():
    console.print(Panel.fit(
        "[bold green]Qwen3-Coder-30B-A3B Local Agent + Free Web Search (v3)[/bold green]\n"
        "• Type [bold yellow]/web <your query>[/bold yellow] to auto-optimize keywords & search Web + News\n"
        "• Type normally for fast offline coding & chat\n"
        "• Type [bold cyan]/clear[/bold cyan] to reset chat memory | [bold red]exit[/bold red] to quit",
        border_style="green"
    ))

    messages = [{"role": "system", "content": get_system_prompt()}]

    while True:
        try:
            user_input = console.input("\n[bold blue]You > [/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break
        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": get_system_prompt()}]
            console.print("[dim]Conversation history cleared.[/dim]")
            continue

        if user_input.lower().startswith("/web "):
            search_query = user_input[5:].strip()
            web_data = fetch_web_context(search_query)
            today = datetime.now().strftime("%A, %B %d, %Y")
            if web_data:
                augmented_prompt = (
                    f"TODAY'S DATE: {today}\n"
                    f"LIVE WEB & NEWS SEARCH RESULTS:\n\n"
                    f"{web_data}\n\n"
                    f"USER QUESTION: {search_query}\n"
                    f"Instructions: Answer the question directly using the search results above. "
                    f"Pay close attention to dates and the SEARCH SUMMARY SNIPPET. "
                    f"Only provide code if the user's question is about programming."
                )
            else:
                augmented_prompt = search_query
            messages.append({"role": "user", "content": augmented_prompt})
        else:
            messages.append({"role": "user", "content": user_input})

        console.print("\n[bold magenta]Qwen3 > [/bold magenta]", end="")
        full_reply = ""
        try:
            stream = client.chat.completions.create(
                model="qwen3-coder",
                messages=messages,
                temperature=0.6,
                top_p=0.8,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                sys.stdout.write(delta)
                sys.stdout.flush()
                full_reply += delta
            print()
            messages.append({"role": "assistant", "content": full_reply})
        except Exception as e:
            console.print(f"\n[bold red]Connection Failed:[/bold red] Make sure windows_start.bat is running!\nError: {e}")
            messages.pop()

if __name__ == "__main__":
    main()