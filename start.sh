#	!!! MAKE APPROPRIATE CHANGES TO  THE PATH !!! 


cd /home/hduser/Desktop/ColorCount
echo " COMPILATION IN PROGRESS !!!"
javac -classpath /usr/local/hadoop/share/hadoop/common/hadoop-common-2.6.4.jar:/usr/local/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-client-core-2.6.4.jar:/usr/local/hadoop/share/hadoop/common/lib/commons-cli-1.2.jar -d /home/hduser/Desktop/ColorCount $1.java
echo " "
echo " CREATING DIRECTORY "
echo " "
rm -rf c || mkdir c
mkdir c || echo "directory  c exists"
echo " "
echo "MOVING .Class FILES TO THE CREATED DIRECTORY"
mv *.class c
echo " "

cd /home/hduser/Desktop/ColorCount
echo " "
echo "CREATING JAR FILE"
echo " "
jar -cvf $1.jar -C /home/hduser/Desktop/ColorCount/c .
echo " "
echo " "
echo " "
echo "DELETING EXISTING OUTPUT FILES"
# hadoop fs -rm -r -skipTrash /user/hduser/Output || echo " unable to delete Output folder"

hadoop fs -rm -r -skipTrash /Output || echo "unable to delete /Output folder"
echo " "
echo " "
echo " "
echo "STARTING TO EXECUTE !!"
cd /usr/local/hadoop
echo " "
bin/hadoop jar /home/hduser/Desktop/ColorCount/$1.jar $1 /Input /Output 
echo " "
echo " "
