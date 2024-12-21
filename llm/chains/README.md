

## LLMChain
Note LLMChain implements the standard Runnable Interface. 🏃

- https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html
- https://api.python.langchain.com/en/latest/chains/langchain.chains.llm.LLMChain.html

It use all the document which may easily exceed the token limitation

## ConversationChain
- https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversation.base.ConversationChain.html
- https://api.python.langchain.com/en/latest/chains/langchain.chains.conversation.base.ConversationChain.html
- https://python.langchain.com/v0.1/docs/modules/memory/conversational_customization/

Chain to have a conversation and load context from memory.

> Deprecated since version 0.2.7: Use RunnableWithMessageHistory: https://python.langchain.com/v0.2/api_reference/core/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html instead.
    
RunnableWithMessageHistory offers several benefits, including:
- Stream, batch, and async support;
- More flexible memory handling, including the ability to manage memory outside the chain;
- Support for multiple threads.

## load_qa_chain
- https://python.langchain.com/api_reference/langchain/chains/langchain.chains.question_answering.chain.load_qa_chain.html
- https://api.python.langchain.com/en/latest/chains/langchain.chains.question_answering.chain.load_qa_chain.html

Load question answering chain.

> Deprecated since version 0.2.13: This class is deprecated. See the following migration guides for replacements based on chain_type:

- stuff: https://python.langchain.com/v0.2/docs/versions/migrating_chains/stuff_docs_chain 
- map_reduce: https://python.langchain.com/v0.2/docs/versions/migrating_chains/map_reduce_chain 
- refine: https://python.langchain.com/v0.2/docs/versions/migrating_chains/refine_chain 
- map_rerank: https://python.langchain.com/v0.2/docs/versions/migrating_chains/map_rerank_docs_chain

## RetrievalQA
- https://python.langchain.com/api_reference/langchain/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html

> Deprecated since version 0.1.17: This class is deprecated. Use the create_retrieval_chain constructor instead. See migration guide here: https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa/ It will be removed in None==1.0.

## ConversationalRetrievalChain
- https://python.langchain.com/api_reference/langchain/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html

ConversationalRetrievalChain = RetrievalChain + ConversationBufferMemory

> Deprecated since version 0.1.17: Use create_history_aware_retriever together with create_retrieval_chain (see example in docstring)() instead. It will be removed in None==1.0.

