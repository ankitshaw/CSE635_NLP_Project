import parlai.core.agents as agents
import parlai.utils.logging as logging
import parlai.tasks.wizard_of_wikipedia.agents as wow_agents

# Load the transformer/generator model
model_path = 'zoo:wikipedia_full/tfidf_retriever/model'
model_agent = agents.create_agent_from_model_file(model_path, {'model':"tfidf_retriever",'optimizer':'adam'})
# model_opt = {'task': 'wizard_of_wikipedia:generator'}
# model_agent.opt.update(model_opt)

# # Load the Wizard of Wikipedia task
# task_name = 'wizard_of_wikipedia:generator'
# task_opt = {'datapath': 'data', 'task': task_name}
# task_agent = wow_agents.WoWGeneratorAgent(opt=task_opt)

# Start a conversation with the model
input_text = 'Tell me about the history of New York City.'
while True:
    # Generate a response from the model
    model_response = model_agent.respond(input_text)
    logging.info(f'Model response: {model_response}')

    # # Generate a response from the task agent
    # task_response = task_agent.act(model_response)
    # logging.info(f'Task response: {task_response["text"]}')

    # Get the next input from the user
    input_text = input('You: ')