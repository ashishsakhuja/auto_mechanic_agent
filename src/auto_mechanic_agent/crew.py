# crew.py

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
import logging
from auto_mechanic_agent.tools.custom_tool import SQLManualTool, ManualQATool, PartsScraperTool
from knowledge.vehicle_knowledge_source import ManualIndex

load_dotenv()

@CrewBase
class AutoMechanicAgent:
    """AutoMechanicAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # register the SQL lookup tool globally
    tools = [
        SQLManualTool(),
        PartsScraperTool(),
    ]

    def __init__(self):
        super().__init__()
        load_dotenv()
        logging.basicConfig(level=logging.INFO)

        # build a filesystem index of all the Toyota manuals
        self.manual_index = ManualIndex(
            manuals_dir="knowledge/manuals/Toyota",
            index_dir="knowledge/manual_index"
        )

    # ───────────────────────────── Agents ──────────────────────────────

    @agent
    def text_parser(self) -> Agent:
        """Cleans up the user’s problem into a concise summary"""
        return Agent(
            config=self.agents_config["text_parser"],
            verbose=True,
        )

    @agent
    def manual_sql(self) -> Agent:
        """Queries the DuckDB to find the PDF path for a given make/model/year"""
        return Agent(
            config=self.agents_config["manual_sql"],
            tools=[SQLManualTool()],
            verbose=True,
        )

    @agent
    def manual_qa_agent(self) -> Agent:
        """Loads and chunks one PDF, runs RetrievalQA over it"""
        qa_tool = ManualQATool(
            manual_index=self.manual_index,
            chunk_size=800,
            chunk_overlap=100,
            top_k=4,
            model_name="gpt-4o-mini",
            temperature=0.0,
        )
        return Agent(
            config=self.agents_config["manual_qa_agent"],
            tools=[qa_tool],
            verbose=True,
        )

    @agent
    def mechanic_expert(self) -> Agent:
        """Provides expert advice on car issues (without manual lookup)"""
        return Agent(
            config=self.agents_config["mechanic_expert"],
            tools=[],
            verbose=True,
        )

    @agent
    def mechanic_supervisor(self) -> Agent:
        """Supervises the mechanic expert and ensures proper solution generation"""
        return Agent(
            config=self.agents_config["mechanic_supervisor"],
            tools=[],
            verbose=True,
        )

    @agent
    def formatter_agent(self) -> Agent:
        """Formats the solution into a PDF-friendly format"""
        return Agent(
            config=self.agents_config["formatter_agent"],
            verbose=True,
        )

    @agent
    def parts_scraper_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["parts_scraper_agent"],
            tools=[PartsScraperTool()],
            verbose=True,
        )


    # ───────────────────────────── Tasks ──────────────────────────────

    @task
    def parse_problem_task(self) -> Task:
        return Task(
            config=self.tasks_config["parse_problem_task"],
        )

    @task
    def find_manual_sql_task(self) -> Task:
        return Task(
            config=self.tasks_config["find_manual_sql_task"],
            tools=[SQLManualTool()],
        )

    @task
    def lookup_manual_task(self) -> Task:
        return Task(
            config=self.tasks_config["lookup_manual_task"],
            tools=[
                SQLManualTool(),
                ManualQATool(
                    manual_index=self.manual_index,
                    chunk_size=800,
                    chunk_overlap=100,
                    top_k=4,
                    model_name="gpt-4.1-mini",
                    temperature=0.0,
                ),
            ],
        )

    @task
    def generate_solution_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_solution_task"],
        )

    @task
    def scrape_parts_task(self) -> Task:
        return Task(
            config=self.tasks_config["scrape_parts_task"],
            tools=[PartsScraperTool()],
        )

    @task
    def enrichment_task(self) -> Task:
        return Task(
            config=self.tasks_config["enrichment_task"],
        )

    @task
    def format_guide_task(self) -> Task:
        return Task(
            config=self.tasks_config["format_guide_task"],
        )

    # ───────────────────────────── Crew Definition ──────────────────────────────

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.text_parser(),
                self.manual_sql(),
                self.manual_qa_agent(),
                self.mechanic_expert(),
                self.mechanic_supervisor(),
                self.parts_scraper_agent(),
                self.formatter_agent(),
            ],
            tasks=[
                self.parse_problem_task(),
                self.find_manual_sql_task(),
                self.lookup_manual_task(),
                self.generate_solution_task(),
                self.scrape_parts_task(),
                self.enrichment_task(),
                self.format_guide_task(),
            ],
            process=Process.sequential,
            tools=[
                SQLManualTool(),
                ManualQATool(
                    manual_index=self.manual_index,
                    chunk_size=800,
                    chunk_overlap=100,
                    top_k=4,
                    model_name="gpt-4o-mini",
                    temperature=0.0,
                ),
            ],
            verbose=True,
        )
