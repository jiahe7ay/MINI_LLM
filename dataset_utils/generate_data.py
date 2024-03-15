import pyarrow.parquet as pq
import pyarrow as pa
import ujson
import numpy as np
from rich import progress
import ujson
from unicodedata import normalize


def split_txt_cropus_to_chunk_data(texts: list, batch_size: int=512 ** 2, max_len: int=512, window_size: int = 2) -> list:
    
    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_txt = ''.join(buffer)
            
            # - window_size为滑动窗口，这样每个窗口都包含有window_size个上文
            for i in range(0, len(buffer_txt), max_len - window_size):

                chunk_data.append(''.join(buffer_txt[i: i + max_len]))
            
            buffer, buffer_len = [], 0
    
    return chunk_data

def gen_wiki(origin_file, output_file):
	liness = []
	with open(origin_file, 'r', encoding='utf-8') as f:
	    lines = f.readlines()
	items, content = [], []
	key_word, kw_line_idx = '', 0
	content_start = False  # 词条内容开始标记

	eos_token= '<|im_end|>'
	for i, line in enumerate(lines):
	    
	    line_strip = line.strip()

	    # 词条以冒号`：`结尾
	    if len(line_strip) > 0 and line_strip[-1] in (':', '：'):
	        key_word = ''.join(line_strip[: -1])
	        kw_line_idx = i 
	        continue
	    
	    # 词条key_word在下一行，则合并上个词条并保存
	    if i == kw_line_idx + 1 and key_word in line_strip or i == len(lines) - 1:
	        txt = ''.join(content)

	        if len(txt) > 0:
	            items.append(f"{txt}{eos_token}")
	            
	        content = []
	        content.append(f"{key_word}：")
	    
	    content.append(line)
	chunk_data=split_txt_cropus_to_chunk_data(items)
	tb=pa.Table.from_arrays([pa.array(chunk_data)],names=['text'])
	pq.write_table(table=tb, where=output_file, row_group_size=50000, data_page_size=50000, )

def process_none(s: str) -> str:
    if s: return s
    return ''


def gen_baike(origin_file):
	baike_items = []
	eos_token = '<|im_end|>'
	max_len = 512
	batch_size, batch_cnt = 2000000, 0
	with open(origin_file, 'r', encoding='utf-8') as f:
		while True:
			line = f.readline()
			if not line: break

			item = ujson.loads(line)
			cur_txt, cur_len = [], 0

			if not item['title']: continue

			temp_txt = f"{item['title']}：{process_none(item['summary'])}"

			cur_len += len(temp_txt)
			cur_txt.append(temp_txt)

			for section in item['sections']:

			    # 太长的截断不要了
			    if cur_len > max_len:
			        break
			    
			    title = f"{section['title']}：" if section['title'] else ""
			    temp_txt = f"{title}{process_none(section['content'])}"
			    
			    cur_len += len(temp_txt)
			    cur_txt.append(temp_txt)
			temp_txt =  normalize('NFKC', ''.join(cur_txt))

			if len(temp_txt) > max_len:
			    # 从 max_len 开始找第一个句号，叹号
			    n, i = len(temp_txt), max_len
			    while i < n and temp_txt[i] not in ('。', '！'):
			        i += 1
			    temp_txt = ''.join(temp_txt[0: i + 1])

	        # 添加 eos token
			temp_txt = f"{temp_txt}{eos_token}"

			baike_items.append( temp_txt )

			if len(baike_items) % batch_size == 0:

				chunk_data = split_txt_cropus_to_chunk_data(baike_items)
				tb = pa.Table.from_arrays([chunk_data], names=['text'])

				file_name = f'../datasets/baike_chunk_512_5.6M_{batch_cnt}.parquet'
				pq.write_table(table=tb, where=file_name, row_group_size=50000, )

				print(f"save to {file_name}")

				batch_cnt += 1
				baike_items = []


# def gen_sky(origin_file, output_file):
# 	lines=[]
# 	with open(origin_file,'r',encoding='utf-8') as f:
# 		for line in f:
# 			item=ujson.loads(line)
# 			lines.append(item['text']+'<|im_end|>')
# 	chunk_data=split_txt_cropus_to_chunk_data(lines)
# 	tb=pa.Table.from_arrays([pa.array(chunk_data)],names=['text'])
# 	pq.write_table(table=tb, where=output_file, row_group_size=50000, data_page_size=50000, )

def gen_sky(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jsonl"):  # 修改为处理JSON Lines文件
            origin_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace('.jsonl', '.parquet'))
            print(f"Processing {origin_file}...")

            lines = []
            with open(origin_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = ujson.loads(line)
                    lines.append(item['text'] + '')  # 确保每行都是一个有效的JSON对象

            if lines:  # 确保文件中有内容
                chunk_data = split_txt_cropus_to_chunk_data(lines)
                tb = pa.Table.from_arrays([pa.array(chunk_data)], names=['text'])
                pq.write_table(table=tb, where=output_file, row_group_size=50000, data_page_size=50000)
                print(f"Processed {origin_file} to {output_file}")
            else:
                print(f"No content in {origin_file}. Skipping.")


def gen_wiki_filter(origin_file, output_file):
	lines=[]
	with open(origin_file,'r',encoding='utf-8') as f:
		items=ujson.load(f)
		for item in items:
			lines.append(item['completion']+'<|im_end|>')
	chunk_data=split_txt_cropus_to_chunk_data(lines)
	tb=pa.Table.from_arrays([pa.array(chunk_data)],names=['text'])
	pq.write_table(table=tb, where=output_file, row_group_size=50000, data_page_size=50000, )

def gen_mbvc(origin_file, output_file):
	lines=[]
	with open(origin_file,'r',encoding='utf-8') as f:
		for line in f:
			item=ujson.loads(line)
			paragraphs=item['段落']
			for paragraph in paragraphs:
				content=paragraph['内容']
				lines.append(content+'<|im_end|>')
	chunk_data=split_txt_cropus_to_chunk_data(lines)
	tb=pa.Table.from_arrays([pa.array(chunk_data)],names=['text'])
	pq.write_table(table=tb, where=output_file, row_group_size=50000, data_page_size=50000, )				


def gen_bell():
	train_data = []
	eval_data = []
	eval_size = 10000
	max_len = 512
	root=".."
	with open(root + '/datasets/train_3.5M_CN.json', 'r', encoding='utf-8') as f:
	    for line in f:
	        item = ujson.loads(line)

	        if len(item['conversations']) != 2: continue

	        conversation = item['conversations']
	        txt = ''
	        if conversation[0]['from'] =='human':
	            txt = f"{conversation[0]['value']}\n{conversation[1]['value']}"
	        else:
	            txt = f"{conversation[1]['value']}\n{conversation[0]['value']}"
	        
	         # 收集测试数据
	        if len(txt) >= max_len and len(txt) < max_len + 8 and len(eval_data) < eval_size and np.random.rand() <= 0.12:
	            eval_data.append(txt)
	            continue
	            

	        if len(txt) >= max_len: continue
	        train_data.append(txt)
	for file in [root + '/datasets/train_2M_CN.json',  root + '/datasets/Belle_open_source_1M.json']:
	    with open(file, 'r', encoding='utf-8') as f:
	        for line in f:
	            item = ujson.loads(line)

	            if item['input'].strip() != '':
	                txt = f"{item['instruction']}\n{item['input']}\n{item['output']}"
	            else:
	                txt = f"{item['instruction']}\n{item['output']}"

	            # 收集测试数据
	            if len(txt) >= max_len and len(txt) < max_len + 8 and len(eval_data) < eval_size and np.random.rand() > 0.75:
	                eval_data.append(txt)
	                continue
	            
	            if len(txt) == 0 or len(txt) >= max_len: continue
	            train_data.append(
	                    txt
	            )
	tb = pa.Table.from_arrays([train_data], names=['text'])
# compression='GZIP'
	pq.write_table(table=tb, where=f'../datasets/bell_pretrain_{max_len}_3M.parquet', row_group_size=20480, data_page_size=20480, )

	tb = pa.Table.from_arrays([eval_data], names=['text'])
# compression='GZIP'
	pq.write_table(table=tb, where=f'../datasets/pretrain_eval_{max_len}_1w.parquet', row_group_size=20480, data_page_size=20480, )


def gen_bell_sft(origin_file,output_file):
	lines=[]
	with open(origin_file,'r',encoding='utf-8') as f:
		for line in f:
			item=ujson.loads(line)
			txt=f"{item['instruction']}{item['output']}"
			if len(txt)==0 or len(txt)>512:
				continue
			lines.append(item)
	tb=pa.Table.from_pylist(lines)
	pq.write_table(table=tb, where=output_file, row_group_size=20480, data_page_size=20480, )

def gen_aplca_sft(origin_file,output_file):
	lines=[]
	with open(origin_file,'r',encoding='utf-8') as f:
		items=ujson.load(f)

		for item in items:
			if 'output' not in item.keys():
				continue
			txt=f"{item['instruction']}{item['output']}"
			if len(txt)==0 or len(txt)>512:
				continue
			lines.append(item)
	#print(lines[0])
	tb=pa.Table.from_pylist(lines)
	pq.write_table(table=tb, where=output_file, row_group_size=20480, data_page_size=20480, )
	

#原本的gen_sky 需要复制多个，没办法读取一个文件夹. 新的gen_sky只需要输入文件夹和输出文件夹的路径即可。并且原本的也会自动修改为.parquet结尾.（喵德注释）
#gen_sky_for_folder("/home/miaode/MINI_LLM/data/SkyPile-150B/data_folder","/home/miaode/MINI_LLM/datasets" )

#这个在readme 没有说清楚是要下载哪一个记得是self_cognition.json 。（喵德注释)
# https://github.com/hiyouga/ChatGLM-Efficient-Tuning/blob/main/data/self_cognition.json
#gen_aplca_sft("../../../datasets/self_cognition.json","../datasets/aplca3.parquet")

gen_bell_sft("../../../datasets/train_2M_CN.json","../datasets/bell3.parquet")
#gen_bell()

#这里的563w_baidubaike要记得解压. 原本download的是7z压缩文件》
#gen_baike('../datasets/563w_baidubaike.json')
#gen_mbvc("../datasets/oscar_202201.part_0000.jsonl","../datasets/mbvc1.parquet")

