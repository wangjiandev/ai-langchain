import { ChatOpenAI } from '@langchain/openai'
import { PromptTemplate, SystemMessagePromptTemplate } from '@langchain/core/prompts'
import { HumanMessagePromptTemplate, ChatPromptTemplate } from '@langchain/core/prompts'
import { CommaSeparatedListOutputParser, StructuredOutputParser } from 'langchain/output_parsers'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { OpenAIEmbeddings } from '@langchain/openai'
import { MemoryVectorStore } from 'langchain/vectorstores/memory'
import { FaissStore } from '@langchain/community/vectorstores/faiss'
import faiss from 'faiss-node'
import { MultiQueryRetriever } from 'langchain/retrievers/multi_query'
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract'
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression'

const main = async () => {
    process.env.LANGCHAIN_VERBOSE = 'true'

    const llm = new ChatOpenAI({
        model: 'qwen3-235b-a22b',
        configuration: {
            baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            apiKey: process.env.DASHSCOPE_API_KEY,
        },
    })

    const embeddings = new OpenAIEmbeddings({
        model: 'text-embedding-v3',
        configuration: {
            baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            apiKey: process.env.DASHSCOPE_API_KEY,
        },
    })

    const directory = './db/bailuyuan'
    const vectorstore = await FaissStore.load(directory, embeddings)
    const compressor = LLMChainExtractor.fromLLM(llm)
    const retriever = new ContextualCompressionRetriever({
        baseCompressor: compressor,
        baseRetriever: vectorstore.asRetriever(2),
    })
    const result = await retriever.invoke('白嘉轩第六个妻子姓什么?')
    console.log(result)
}

main()
