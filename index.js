import { OpenAI } from 'langchain/llms';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as dotenv from 'dotenv';

dotenv.config();

const txtFileName = 'radical_condor';
const question = 'What is radical condor?';
const txtPath = `./${txtFileName}.txt`;
const VECTOR_STORE_PATH = `${txtFileName}.index`;

export const run = async () => {
	const model = new OpenAI({});

	let vectorStore;
	if (fs.existsSync(VECTOR_STORE_PATH)) {
		vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
	} else {
		const text = fs.readFileSync(txtPath, 'utf8');
		const textSplitter = new RecursiveCharacterTextSplitter({
			chunkSize: 1000,
		});
		const docs = await textSplitter.createDocuments([text]);
		vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

		await vectorStore.save(VECTOR_STORE_PATH);
	}

	const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

	const res = await chain.call({
		query: question,
	});

	console.log({ res });
};

run();
