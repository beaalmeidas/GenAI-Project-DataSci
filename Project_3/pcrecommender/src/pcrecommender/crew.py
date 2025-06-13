from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

from crewai_tools import SerperDevTool




@CrewBase
class Pcrecommender():
    """Pcrecommender crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def computer_specialist(self) -> Agent:
        serper_tool = SerperDevTool()
        return Agent(
            config=self.agents_config['computer_specialist'], # type: ignore[index]
            verbose=True,
            tools=[serper_tool]
        )

    @agent
    def price_researcher(self) -> Agent:
        serper_tool = SerperDevTool()
        return Agent(
            config=self.agents_config['price_researcher'], # type: ignore[index]
            verbose=True,
            tools=[serper_tool]
        )
        
    @agent
    def usage_recommender(self) -> Agent:
        return Agent(
            config=self.agents_config['usage_recommender'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def define_computer_config_task(self) -> Task:
        return Task(
            config=self.tasks_config['define_computer_config_task'], # type: ignore[index]
        )

    @task
    def search_prices_task(self) -> Task:
        return Task(
            config=self.tasks_config['search_prices_task'], # type: ignore[index]
            
        )
        
    @task
    def suggest_usage_task(self) -> Task:
        return Task(
            config=self.tasks_config['suggest_usage_task'], # type: ignore[index]
            
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Pcrecommender crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
