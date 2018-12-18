import java.io.IOException;
import java.util.*;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.Mapper;


import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class ColorCount{
	
	
	//mapper
	public static class color_mapper extends Mapper<LongWritable,Text,Text,IntWritable>
{
	//@Override
	public void map(LongWritable key,Text value,Context context)
	throws IOException,InterruptedException{
		
		StringTokenizer iterator=new StringTokenizer(value.toString(),",");
		while(iterator.hasMoreTokens())
		{
			value.set(iterator.nextToken());
			context.write(value,new IntWritable(1));
		}

	}
}

//reducer
public static class color_reducer 
	extends Reducer<Text,IntWritable,IntWritable,Text>{
//@Override
public void reduce(Text key,Iterable<IntWritable> values,Context context)
throws IOException,InterruptedException{
	int sum=0;
	for(IntWritable value:values)
		{
			sum=sum+value.get();
		}
		Text x=new Text();
		x.set(key);
		context.write(new IntWritable(sum),x);
	}
}



//main

	public static void main (String args[])
	throws Exception{
		if(args.length!=2)
		{
			System.err.println("useage output.txt <input path> <output path>");
			System.exit(-1);
		}
		Job job=new Job();
		job.setJarByClass(ColorCount.class);
		job.setJobName("COLOR COUNT");

		FileInputFormat.addInputPath(job,new Path(args[0]));
		FileOutputFormat.setOutputPath(job,new Path(args[1]));

		job.setMapperClass(color_mapper.class);
		job.setReducerClass(color_reducer.class);

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(IntWritable.class);

		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		System.exit(job.waitForCompletion(true)?0:1);

	}


}
