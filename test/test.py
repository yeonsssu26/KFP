import torch
import numpy as np
import time
import itertools

from utils import utils
from data import preprocess
from train import train

def test_sentence_one():
    device = utils.GPU_setting()
    _, test_dataloader, _ = preprocess()
    model, tokenizer = train.save()

    utils.run_test(model, test_dataloader, device)

    sentence = "Whether audiences will get behind The Lightning Thief is hard to predict. Overall, it's an entertaining introduction to a promising new world -- but will the consuming shadow of Potter be too big to break free of?"
    logits = utils.test_sentence_unit(model, device, tokenizer, [sentence])

    print(logits)
    print(np.argmax(logits))

    start_time = time.time()
    sentence = "i hate you"
    logits = utils.test_sentence_unit(model, device, tokenizer, [sentence])

    print(logits)
    print(np.argmax(logits))
    print("  Loading took: {:}".format(utils.format_time(time.time() - start_time)))

def test_sentence_many(model, device, tokenizer, sentences):
    start_time = time.time()
    
    # 출력된 label 리스트
    label_list = list() 
    
    # 평가모드로 변경
    model.eval()

    # 문장을 입력 데이터로 변환
    inputs, masks = utils.convert_input_data(tokenizer, sentences)

    # 데이터를 GPU에 넣음
    b_input_ids = inputs.to(device)
    b_input_mask = masks.to(device)
    
    test_data = utils.TensorDataset(b_input_ids, b_input_mask)
    test_dataloader = utils.DataLoader(test_data, batch_size=32)
    
    # 데이터로더에서 배치만큼 반복하여 가져옴
    for step, batch in enumerate(test_dataloader):
        # 경과 정보 표시
        if step % 100 == 0 and not step == 0:
            elapsed = utils.format_time(time.time() - start_time)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(test_dataloader), elapsed))

        # 배치를 GPU에 넣음
        batch = tuple(b.to(device) for b in batch)

        # 배치에서 데이터 추출
        b_input_ids, b_input_mask = batch
    
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)

        # 로스 구함
        logits = outputs[0]

        # CPU로 데이터 이동
        preds = logits.detach().cpu().numpy()
        pred_flat = np.argmax(preds, axis=1).flatten()
        label_list.append(list(pred_flat))
    
    # 이중 리스트를 단일 리스트로 변경
    result = list(itertools.chain.from_iterable(label_list))    
    return result
