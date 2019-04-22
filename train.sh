if [ $1 = "offsets" ]; then
   for i in `seq $2`
   do
       python offsets_train.py --model_id $((i - 1)) --n_bertlayers 22 --train_batch_size 4 --gradient_accumulation_steps 5 --save_model --log_dir "log/" --data_dir "../" --model_dir "model/"
   done

elif [ $1 = "nli" ]; then
   for i in `seq $2`
   do
       python nli_train.py --model_id $((i - 1)) --n_bertlayers 22 --train_batch_size 4 --gradient_accumulation_steps 5 --save_model --max_len 200 --no_pooler --log_dir "log/" --data_dir "../" --model_dir "model/"
   done
fi    
