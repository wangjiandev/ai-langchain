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

const main = async () => {
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

    const loader = new TextLoader('bai.txt')
    const docs = await loader.load()

    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 200,
        chunkOverlap: 20,
    })

    const splitDocs = await splitter.splitDocuments(docs)
    const vectorStore = new FaissStore(embeddings, {})
    const batchSize = 10
    for (let i = 0; i < splitDocs.length; i += batchSize) {
        const batch = splitDocs.slice(i, i + batchSize)
        await vectorStore.addDocuments(batch)
    }
    const directory = './db/bailuyuan'
    await vectorStore.save(directory)
}

main()
