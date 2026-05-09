"""CLI interface for CV QA Agent"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table


logger = logging.getLogger(__name__)
console = Console()


class CVQACLI:
    """Command-line interface for CV Question Answering"""
    
    def __init__(self, agent, memory, settings):
        """
        Initialize CLI
        
        Args:
            agent: CVQAAgent instance
            memory: ConversationMemory instance
            settings: Settings instance
        """
        self.agent = agent
        self.memory = memory
        self.settings = settings
        
        logger.info("Initialized CVQACLI")
    
    def display_welcome(self):
        """Display welcome message"""
        welcome_text = """
# CV Question Answering Agent

Welcome! I can answer questions about the CV you've provided.

**Commands:**
- Type your question and press Enter
- Type 'history' to see conversation history
- Type 'stats' to see session statistics
- Type 'clear' to clear conversation history
- Type 'exit' or 'quit' to end the session

**Tips:**
- Ask specific questions about work experience, education, skills, etc.
- I can handle follow-up questions using conversation context
- All answers are strictly based on the CV content
        """
        
        console.print(Panel(Markdown(welcome_text), title="🤖 CV QA Agent", border_style="blue"))
        console.print()
    
    def display_config(self):
        """Display current configuration"""
        table = Table(title="Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("LLM Provider", self.settings.llm_provider)
        table.add_row("Model", self.settings.model_name)
        table.add_row("CV File", self.settings.cv_file_path)
        table.add_row("Temperature", str(self.settings.temperature))
        table.add_row("Retrieval Top-K", str(self.settings.retrieval_top_k))
        
        console.print(table)
        console.print()
    
    def run(self):
        """Run the interactive CLI"""
        try:
            self.display_welcome()
            self.display_config()
            
            while True:
                try:
                    # Get user input
                    console.print()
                    question = Prompt.ask("[bold cyan]Your question[/bold cyan]")
                    
                    if not question.strip():
                        continue
                    
                    # Handle commands
                    command = question.strip().lower()
                    
                    if command in ['exit', 'quit', 'q']:
                        self._handle_exit()
                        break
                    elif command == 'history':
                        self._display_history()
                        continue
                    elif command == 'stats':
                        self._display_stats()
                        continue
                    elif command == 'clear':
                        self._clear_history()
                        continue
                    elif command == 'help':
                        self.display_welcome()
                        continue
                    
                    # Process question
                    self._process_question(question)
                    
                except KeyboardInterrupt:
                    console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                    continue
                except Exception as e:
                    logger.error(f"Error in CLI loop: {str(e)}")
                    console.print(f"[red]Error: {str(e)}[/red]")
                    continue
        
        except Exception as e:
            logger.error(f"Fatal error in CLI: {str(e)}")
            console.print(f"[red]Fatal error: {str(e)}[/red]")
            sys.exit(1)
    
    def _process_question(self, question: str):
        """Process a user question"""
        try:
            # Get chat history for context
            recent_history = self.memory.get_recent_history(3)
            chat_history = []
            for exchange in recent_history:
                chat_history.append(("human", exchange['question']))
                chat_history.append(("ai", exchange['answer']))
            
            # Query agent
            result = self.agent.query(question, chat_history)
            
            answer = result['answer']
            retrieved_context = result.get('retrieved_context', '')
            is_grounded = result.get('is_grounded', True)
            
            # Display answer - clean and simple
            console.print()
            console.print(f"[bold green]Answer:[/bold green] {answer}")
            console.print()
            
            # Add to memory
            self.memory.add_exchange(
                question=question,
                answer=answer,
                retrieved_context=retrieved_context,
                metadata={
                    'is_grounded': is_grounded,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            logger.info(f"Processed question successfully: '{question[:50]}...'")
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            console.print(f"[red]Error processing question: {str(e)}[/red]")
    
    def _display_history(self):
        """Display conversation history"""
        history = self.memory.get_full_history()
        
        if not history:
            console.print("[yellow]No conversation history yet.[/yellow]")
            return
        
        console.print()
        console.print(Panel("📜 Conversation History", border_style="blue"))
        
        for i, exchange in enumerate(history, 1):
            console.print(f"\n[bold cyan]Q{i}:[/bold cyan] {exchange['question']}")
            console.print(f"[bold green]A{i}:[/bold green] {exchange['answer'][:200]}...")
            console.print(f"[dim]Time: {exchange['timestamp']}[/dim]")
    
    def _display_stats(self):
        """Display session statistics"""
        stats = self.memory.get_stats()
        
        table = Table(title="Session Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Questions", str(stats['total_exchanges']))
        table.add_row("Session Start", stats['session_start'])
        table.add_row("Duration (minutes)", f"{stats['session_duration_minutes']:.1f}")
        
        console.print()
        console.print(table)
    
    def _clear_history(self):
        """Clear conversation history"""
        self.memory.clear()
        console.print("[green]✓ Conversation history cleared.[/green]")
    
    def _handle_exit(self):
        """Handle exit command"""
        console.print("\n[yellow]Saving conversation history...[/yellow]")
        
        try:
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(self.settings.output_dir) / f"conversation_{timestamp}.txt"
            
            # Export conversation
            self.memory.export_to_file(str(output_file))
            
            console.print(f"[green]✓ Conversation saved to: {output_file}[/green]")
            console.print("\n[bold blue]Thank you for using CV QA Agent! Goodbye! 👋[/bold blue]\n")
            
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            console.print(f"[red]Error saving conversation: {str(e)}[/red]")


