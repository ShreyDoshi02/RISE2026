import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_community.agent_toolkits.load_tools import load_tools

# 1. Load environment variables
load_dotenv()

def main():
    try:
        # Load the Groq API Key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            print("Error: Please set your GROQ_API_KEY in the .env file.")
            print("You can get an API key from https://console.groq.com/keys")
            return

        print("Initializing Groq LLM...")
        # 2. Initialize Groq LLM with Llama3 model
        llm = ChatGroq(
            temperature=0,
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key
        )
        
        # 4. Set up the Calculator Tool
        print("Setting up tools...")
        import numexpr
        def calculate(expression: str) -> str:
            """Evaluates a mathematical expression and returns the result."""
            try:
                return str(numexpr.evaluate(expression.strip()).item())
            except Exception as e:
                return f"Error evaluating expression: {e}"
        
        calc_tool = Tool(
            name="Calculator",
            func=calculate,
            description="Useful for evaluating mathematical expressions. Input should be a math expression like '347 * 892'."
        )
        
        # Combine our custom tools
        tools = [calc_tool]

        # 5. Initialize the Agent
        print("Initializing Agent (Modern create_agent due to Python 3.14 compat issues)...")
        from langchain.agents import create_agent
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt="You are a helpful assistant capable of mathematics."
        )

        # 6. Execute Queries
        print("\n--- Starting Interactive Agent (type 'exit' or 'quit' to stop) ---\n")
        
        while True:
            query = input("> Enter your query: ").strip()
            if query.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            
            if not query:
                continue

            try:
                # Run the modern agent using invoke
                response = agent.invoke({"messages": [{"role": "user", "content": query}]})
                
                final_message = response["messages"][-1].content
                print(f"\nFinal Answer: {final_message}\n")
                print("-" * 50 + "\n")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"\nError during agent execution for query '{query}': {e}\n")
                print("-" * 50 + "\n")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fatal Initialization Error: {e}")

if __name__ == "__main__":
    main()