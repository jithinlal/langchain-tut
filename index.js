import { OpenAI ,OpenAIEmbeddings} from '@langchain/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import * as fs from 'fs';
import * as dotenv from 'dotenv';
import select, { Separator } from '@inquirer/select';
import input from '@inquirer/input';

dotenv.config();

const txtFileName = 'radical_condor';
const txtPath = `./${txtFileName}.txt`;
const VECTOR_STORE_PATH = `${txtFileName}.index`;

export const run = async (question) => {
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

	const res = await chain.invoke({
		query: question,
	});

	console.log("----------")
	console.log(res.text);
	console.log("----------")
};



(async function runProgram(){
	while(true) {
		const answer = await select({
			message: 'select an option from below',
			choices: [
				{
					name: 'ask a question',
					value: 'question',
					description: 'Ask a question regarding your document',
				},
				{
					name: 'quit the cycle',
					value: 'quit',
					description: 'quit the program',
				},
				new Separator(),
			],
		});

		if(answer === "question") {
			const question = await input({ message: 'Ask your question' });
			await run(question)
		} else {
			process.exit(0)
		}
	}
})()