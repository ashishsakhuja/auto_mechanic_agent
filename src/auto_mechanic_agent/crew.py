from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
import logging
from auto_mechanic_agent.tools.custom_tool import PDFCreatorTool
from auto_mechanic_agent.tools.custom_tool import ImageGenTool
from auto_mechanic_agent.tools.custom_tool import QueryManifestTool

load_dotenv()


@CrewBase
class AutoMechanicAgent():
    """AutoMechanicAgent crew"""

    agents: List[BaseAgent]
    tasks: List[Task]
    tools = [PDFCreatorTool(), ImageGenTool(), QueryManifestTool()]


    def __init__(self):
        super().__init__()
        load_dotenv()
        logging.basicConfig(level=logging.INFO)

    @agent
    def text_parser(self) -> Agent:
        """Cleans up the user’s problem into a concise summary"""
        return Agent(
            config=self.agents_config["text_parser"],
            verbose=True,
        )

    @agent
    def mechanic_expert(self) -> Agent:
        """Provides expert advice on car issues"""
        return Agent(
            config=self.agents_config["mechanic_expert"],
            tools=[QueryManifestTool()],
            verbose=True,
        )

    @agent
    def pdf_creator(self) -> Agent:
        """Renders HTML into a PDF file"""
        return Agent(
            config=self.agents_config["pdf_creator"],
            verbose=True,
        )

    @task
    def parse_problem_task(self) -> Task:
        return Task(
            config=self.tasks_config["parse_problem_task"],
        )

    @task
    def generate_solution_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_solution_task"],
        )

    @task
    def format_for_pdf_task(self) -> Task:
        return Task(
            config=self.tasks_config["format_for_pdf_task"],
        )

    @task
    def generate_pdf_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_pdf_task"],
            tools=[PDFCreatorTool()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AutoMechanicAgent crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            tools=[PDFCreatorTool(), QueryManifestTool()],
            verbose=True,
        )
