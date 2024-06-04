# Plan And Execute Agent

This type of Agent can handle complex, multi-level requests by dividing up task and planning 
steps to take sequentially

### Advantages
- More adaptable to complicated queries.
- Save on cost (only the planner is gpt-4o, the rest can be gpt-3.5 because tasks are divided to be modular and easy to execute)

### Drawback
- Many API call -> low speed and not reactive
- Chaining multiple web-search requests takes a lot of time and can be costly

### Potential Upgrade
This has open the door to discuss Multi Agents Systems, where we have 1 planner who delegate works to multiple agents, with their own tools 
and specialties.

This has shown to have promising results. However, this is beyond the scope of our Search Engine project, but it can be useful when 
for more advanced projects in the future.

### Resources

[LangGraph Documentation](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/)

[Video on Multi-Agent System](https://www.youtube.com/watch?v=s-jYxgKMqRc&ab_channel=SamWitteveen)

[Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/abs/2405.02957)

[CrewAI's Hierarchical Process](https://docs.crewai.com/how-to/Hierarchical/)